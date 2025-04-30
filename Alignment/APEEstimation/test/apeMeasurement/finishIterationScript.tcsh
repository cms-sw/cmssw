#!/bin/tcsh

set curDir=$PWD
echo $curDir
cd $1/src

eval `scramv1 runtime -csh`
cd $curDir

python3 $1/src/Alignment/APEEstimation/test/apeMeasurement/mergeStep.py --workingArea $2 --measName $3 --numFiles $4 --iteration $5 --isBaseline $6

cmsRun $1/src/Alignment/APEEstimation/test/apeMeasurement/apeDetermination_cfg.py workingArea=$2 measName=$3 iteration=$5 isBaseline=$6 baselineName=$7 

cmsRun $1/src/Alignment/APEEstimation/test/apeMeasurement/apeWrite_cfg.py workingArea=$2  measName=$3 iteration=$5 isBaseline=$6 

-- dummy change --
