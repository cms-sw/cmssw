#!/bin/tcsh
set curDir=$PWD
echo $curDir
cd $1/src

eval `scramv1 runtime -csh`
cd $curDir

python3 $1/src/Alignment/APEEstimation/test/apeMeasurement/prepareMeasurement.py --workingArea $2  --globalTag $3 --measName $4 --isCosmics $5 --maxIterations $6 --baselineName $7 --dataDir $8 --fileName $9 --maxEvents $10 --isBaseline $11
-- dummy change --
