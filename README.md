# Multiple-Object-Tracking-on-Aerial-Images
Implementation of Detection based and detection less trackers to track the object in successive frames

1. First folder named "Detection based Trackers" uses YOLOv3 as an architecture on PASCAL Datasets.
2. Second Folder Named "Detection Less Trackers" is the implementation of SiamMask for Multiple objects Trackers and the model is being cloned from https://github.com/foolwood/SiamMask repository, to understand the working architecture follow the above link and go to original repo for SiamMask by foolwood

->Implemented Detection based trackers, where worked on centroid tracker and Deep SORT tracker with YOLOv4 as detection algorithm on PASCAL datasets
->Used SiamMask, a semi-supervised state of the art technique as a detection less tracker, to first draw the bounding box around the object from user Input and segment   the video Stream around that bounding box and track the segmented part in successive frames
->Implemented technique to track multiple objects in a frame using SiamMask
