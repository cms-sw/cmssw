#!/bin/bash
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }
REMOTE="/store/group/dpg_tracker_strip/comm_tracker/Strip/Calibration/calibrationtree/GR18"
RUN="325310"
FILE="calibTree_${RUN}.root"
COMMMAND=`xrdfs cms-xrd-global.cern.ch locate ${REMOTE}/${FILE}`
STATUS=$?
echo "xrdfs command status = "$STATUS
if [ $STATUS -eq 0 ]; then
    (cmsRun ${LOCAL_TEST_DIR}/Gains_CT_step1.py firstRun=${RUN} lastRun=${RUN} inputFiles=root://cms-xrd-global.cern.ch//$REMOTE/$FILE outputFile=PromptCalibProdSiStripGainsFromTree.root) || die 'Failure running cmsRun Gains_CT_step1.py firstRun=${RUN} lastRun=${RUN} inputFiles=root://cms-xrd-global.cern.ch//$REMOTE/$FILE outputFile=PromptCalibProdSiStripGainsFromTree.root' $?
    (cmsRun ${LOCAL_TEST_DIR}/Gains_CT_step2.py inputFiles=file:PromptCalibProdSiStripGainsFromTree.root DQMOutput=True) || die 'Failure running cmsRun Gains_CT_step2.py inputFiles=file:PromptCalibProdSiStripGainsFromTree.root DQMOutput=True' $?
else
  die "SKIPPING test, file ${FILE} not found" 0
fi
