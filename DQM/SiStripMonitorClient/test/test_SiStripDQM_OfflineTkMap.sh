#!/bin/bash
function die { echo $1: status $2; exit $2; }
GT=`echo ${@} | python3 -c 'import Configuration.AlCa.autoCond as AC;print(AC.autoCond["run3_data_prompt"])'`
RUN="319176"
DQMFILE="/store/group/comm_dqm/DQMGUI_data/Run2018/ZeroBias/R0003191xx/DQM_V0001_R000319176__ZeroBias__Run2018B-PromptReco-v2__DQMIO.root"
COMMMAND=`xrdfs cms-xrd-global.cern.ch locate $DQMFILE`
STATUS=$?
echo "xrdfs command status = "$STATUS
if [ $STATUS -eq 0 ]; then
    echo "Using file ${DQMFILE} and run ${RUN}. Running in ${SCRAM_TEST_PATH}."
    (cmsRun "${SCRAM_TEST_PATH}/SiStripDQM_OfflineTkMap_Template_cfg_DB.py" globalTag="$GT" runNumber="$RUN" dqmFile=" root://cms-xrd-global.cern.ch//$DQMFILE" detIdInfoFile="file.root") || die 'failed running SiStripDQM_OfflineTkMap_Template_cfg_DB.py' $?
else 
  die "SKIPPING test, file ${DQMFILE} not found" 0
fi
