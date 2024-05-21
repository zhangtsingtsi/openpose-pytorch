import sys
sys.path.append('../')
import os
import cv2
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints


estimator = BodyPoseEstimator(pretrained=True)

video_file = './media/example.mp4'
output_dir = './output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(output_dir, 'output_' + os.path.basename(video_file))

videoclip = cv2.VideoCapture(video_file)
width = int(videoclip.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoclip.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = videoclip.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'X264', 'avc1'
writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

if not writer.isOpened():
    print("Error: VideoWriter not opened.")
    sys.exit()

while videoclip.isOpened():
    flag, frame = videoclip.read()
    if not flag:
        break
    keypoints = estimator(frame)
    frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
    frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)
    
    #cv2.imshow('Video Demo', frame)
    writer.write(frame)
    
    if cv2.waitKey(20) & 0xff == 27: # exit if pressed `ESC`
        break
    
videoclip.release()
writer.release()
cv2.destroyAllWindows()
