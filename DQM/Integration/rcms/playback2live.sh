#/bin/bash

for i in *playback.cfg ;  do
ls $i
sed -i 's/srv-c2d05-18/srv-c2d05-19/' $i 
sed -i 's/dqmdev/dqmpro/' $i > help 
sed -i 's/cmsmon:50082\/urn:xdaq-application:lid=29/srv-c2d05-14.cms:22100\/urn:xdaq-application:lid=30/' $i 
done
