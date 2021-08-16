#!/bin/bash

function die { echo $1: status $2; exit $2; }

GT=`echo ${@} | python3 -c 'import Configuration.AlCa.autoCond as AC;print(AC.autoCond["run3_data_prompt"])'`
FILE=/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2018/ZeroBias/R0003191xx/DQM_V0001_R000319176__ZeroBias__Run2018B-PromptReco-v2__DQMIO.root

echo "using GlobalTag: " $GT
echo "using file: " $FILE
cmsRun ${CMSSW_BASE}/src/DQM/SiStripMonitorClient/test/SiStripDQM_OfflineTkMap_Template_cfg_DB.py globalTag=$GT runNumber=319176 dqmFile=$FILE  detIdInfoFile=file.root  || die 'failed running SiStripDQM_OfflineTkMap_Template_cfg_DB.py' $?
