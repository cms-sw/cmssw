#!/bin/bash
function die { echo $1: status $2; exit $2; }
DQMFILE="/store/group/comm_dqm/DQMGUI_data/Run2018/ZeroBias/R0003191xx/DQM_V0001_R000319176__ZeroBias__Run2018B-PromptReco-v2__DQMIO.root"
COMMMAND=`xrdfs cms-xrd-global.cern.ch locate $DQMFILE`
STATUS=$?
echo "xrdfs command status = "$STATUS
if [ $STATUS -eq 0 ]; then
    echo "Using file ${DQMFILE} Running in ${SCRAM_TEST_PATH}."
    xrdcp root://cms-xrd-global.cern.ch//$DQMFILE DQM_V0001_R000319176__ZeroBias__Run2018B-PromptReco-v2__DQMIO.root 
    (python3 ${SCRAM_TEST_PATH}/../scripts/PhaseITreeProducer.py DQM_V0001_R000319176__ZeroBias__Run2018B-PromptReco-v2__DQMIO.root) || die 'failed running PhaseITreeProducer.py' $?
    (python3 ${SCRAM_TEST_PATH}/../scripts/TH2PolyOfflineMaps.py DQM_V0001_R000319176__ZeroBias__Run2018B-PromptReco-v2__DQMIO.root 3000 2000 ) || die 'failed running TH2PolyOfflineMaps.py' $?
else 
  die "SKIPPING test, file ${DQMFILE} not found" 0
fi
