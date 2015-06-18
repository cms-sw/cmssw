#!/bin/bash

eval `scramv1 r -sh`

tnum=3
DQMSEQUENCE=HARVESTING:validationHarvesting+dqmHarvesting

#cmsDriver.py step3_MC1_1 -s HARVESTING:validationpreprodHarvesting --harvesting AtRunEnd --conditions auto:mc --filetype DQM --filein file:step2_MC1_1_RAW2DIGI_RECO_VALIDATION.root --mc --customise DQMServices/Components/test/customHarvesting.py >& q3.1.log && mv DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root.1

#if [ $? -ne 0 ]; then
#	return 1
#fi

#cmsDriver.py step3_MC1_2 -s ${DQMSEQUENCE} --harvesting AtRunEnd --conditions auto:mc --filetype DQM --filein file:step2_MC1_2_RAW2DIGI_RECO_DQM_inDQM.root --mc --customise DQMServices/Components/test/customHarvesting.py >& q3.2.log && mv DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root.2

#if [ $? -ne 0 ]; then
#	return 1
#fi

#cmsDriver.py step3_MC1_3 -s HARVESTING:validationpreprodHarvesting+dqmHarvestingPOG --harvesting AtRunEnd --conditions auto:mc --filetype DQM --filein file:step2_MC1_3_RAW2DIGI_RECO_VALIDATION_DQM_inDQM.root --mc --customise DQMServices/Components/test/customHarvesting.py >& q3.3.log && mv DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root.3

#if [ $? -ne 0 ]; then
#	return 1
#fi

#python /afs/cern.ch/user/r/rovere/public/checkMem.py cmsDriver.py step3_MC1_4 -s ${DQMSEQUENCE} --harvesting AtRunEnd --conditions auto:mc --filetype DQM --filein file:step2_MC1_4_RAW2DIGI_RECO_VALIDATION_DQM_inDQM.root --mc --customise DQMServices/Components/test/customHarvesting.py >& q3.4.log && mv DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_4.root
cmsDriver.py step3_MC1_4 -s ${DQMSEQUENCE} --harvesting AtRunEnd --conditions auto:run1_mc --filetype DQM --filein file:step2_MC1_4_RAW2DIGI_L1Reco_RECO_VALIDATION_DQM_inDQM.root --mc --customise DQMServices/Components/test/customHarvesting.py --python_filename=test_${tnum}_b_1.py  --no_exec

cmsRun -e test_${tnum}_b_1.py >& q3.4.log

mv DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_4.root


if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_b_1}.xml
