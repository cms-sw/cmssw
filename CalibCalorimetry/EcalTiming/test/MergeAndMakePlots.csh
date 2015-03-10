#!/bin/tcsh -f

set runnum = $1

./mergeJason.sh -r $runnum -at Laser
./makeLaserPlots.sh -r $runnum 
#./makeCalibPlots.sh -r $runnum

echo "DONE"
#end of file
