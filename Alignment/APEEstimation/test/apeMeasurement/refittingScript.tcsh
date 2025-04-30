#!/bin/tcsh

set curDir=$PWD
echo $curDir
cd $1/src
eval `scramv1 runtime -csh`

cd $curDir

xrdcp $2 reco.root

cmsRun $1/src/Alignment/APEEstimation/test/apeMeasurement/refitting_cfg.py workingArea=$3 globalTag=$4 measName=$5 fileNumber=$6 iteration=$7 lastIter=$8 isCosmics=$9 maxEvents=$10

rm reco.root


-- dummy change --
