#!/bin/tcsh -f

set runnum=$1

echo "Launching both laser and calibration analysis for run ${1}"

./launchLaserCRAB.sh -r $runnum -s 50000 -d "/TestEnables/Commissioning09-v1/RAW" -at Laser; 
./launchLaserCRAB.sh -r $runnum -s 20000 -d "/TestEnables/Commissioning09-v1/RAW" -at Calib;

#end of file
