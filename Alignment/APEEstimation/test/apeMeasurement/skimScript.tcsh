#!/bin/tcsh

set curDir=$PWD
echo $curDir
cd $1/src
eval `scramv1 runtime -csh`
cd $curDir

cmsRun $1/src/Alignment/APEEstimation/test/apeMeasurement/skim_cfg.py $2

python3 $1/src/Alignment/APEEstimation/test/apeMeasurement/moveSkimOutput.py -s $curDir -t $3 -f $4
-- dummy change --
