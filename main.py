import cv2
def facebox(facenet,frame):
    frameHeight=frame.shape[0]
    framewidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame,1.0,(227,227),[104,117,123],swapRB=False)
    facenet.setInput(blob)
    detection=facenet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*framewidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*framewidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)

    
    return frame,bboxs







facemodel="opencv_face_detector_uint8.pb"
faceproto="opencv_face_detector.pbtxt"

ageproto="age_deploy.prototxt"
agemodel="age_net.caffemodel"

genderproto="gender_deploy.prototxt"
gendermodel="gender_net.caffemodel"



facenet=cv2.dnn.readNet(facemodel,faceproto)
agenet=cv2.dnn.readNet(agemodel,ageproto)
gendernet=cv2.dnn.readNet(gendermodel,genderproto)


model_mean_values=(78.4263377603,87.7689143744,114.89587746)
agelist=['(0-2)','(4-6)','(8-12)','(15-20)','(25-35)','(38-43)','(48-53)','(60-100)',]
genderlist=['Male','Female']



video=cv2.VideoCapture(0)

while True:
    ret,frame=video.read()
    frame,bboxs=facebox(facenet,frame)
    for bbox in bboxs:
        face=frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        blob=cv2.dnn.blobFromImage(face,1.0,(227,227),model_mean_values,swapRB=False)
        gendernet.setInput(blob)
        genderPred=gendernet.forward()
        gender=genderlist[genderPred[0].argmax()]



        agenet.setInput(blob)
        agePred=agenet.forward()
        age=agelist[agePred[0].argmax()]


        label="{},{}".format(gender,age)
        cv2.rectangle(frame,(bbox[0],bbox[1]-30),(bbox[2],bbox[1]),(0,255,0),-1)
        cv2.putText(frame,label,(bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
        
    cv2.imshow('Age-Gender',frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
