#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo -e "TESTING Pixel Condition plotting codes ..."
echo -e "TESTING SiPixelQualityPlotter...\n\n"
REMOTE="/store/group/comm_luminosity/LumiProducerFromBrilcalc/"
INPUTFILE="LumiData_2018_20200401.csv"
COMMMAND=`xrdfs cms-xrd-global.cern.ch locate ${REMOTE}${DQMFILE}`
STATUS=$?
echo "xrdfs command status = "$STATUS
if [ $STATUS -eq 0 ]; then
    echo "Using file ${INPUTFILE}. Running in ${PWD}."
    xrdcp root://cms-xrd-global.cern.ch/${REMOTE}${INPUTFILE} .
    cmsRun ${SCRAM_TEST_PATH}/SiPixelQualityPlotter_cfg.py inputLumiFile=${INPUTFILE} || die "Failure running SiPixelQualityPlotter_cfg.py" $?
    rm -fr ./${INPUTFILE}
else 
  die "SKIPPING test, file ${INPUTFILE} not found" 0
fi
