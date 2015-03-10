#!/bin/tcsh -f

set runnum=$1

echo "Checking both laser and calibration analysis for run ${1}"

./checkResultsCRAB.sh -r Laser_${runnum}; 
./checkResultsCRAB.sh -r Calib_${runnum};

#end of file
