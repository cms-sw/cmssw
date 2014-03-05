#!/bin/bash

eval `scramv1 r -sh`

tnum=2
step=b
DQMSEQUENCE=HARVESTING:dqmHarvesting

cmsDriver.py test_${tnum}_${step}_1 -s ${DQMSEQUENCE} --conditions auto:com10 --filetype DQM --filein file:test_${tnum}_a_1_RAW2DIGI_RECO_DQM_inDQM.root --data --scenario=cosmics --customise DQMServices/Components/test/customHarvesting.py --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& q2.1.log && mv DQM_V0001_R000142560__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000142560__Global__CMSSW_X_Y_Z__RECO_1.root

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

cmsDriver.py test_${tnum}_${step}_2 -s ${DQMSEQUENCE} --conditions auto:com10 --filetype DQM --filein file:test_${tnum}_a_2_RAW2DIGI_RECO_DQM_inDQM.root --data --scenario=cosmics --customise DQMServices/Components/test/customHarvesting.py --no_exec --python_filename=test_${tnum}_${step}_2.py

cmsRun -e test_${tnum}_${step}_2.py >& q2.2.log && mv DQM_V0001_R000142560__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000142560__Global__CMSSW_X_Y_Z__RECO_2.root

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_2}.xml

cmsDriver.py test_${tnum}_${step}_3 -s ${DQMSEQUENCE} --conditions auto:com10 --filetype DQM --filein file:test_${tnum}_a_3_RAW2DIGI_RECO_DQM_inDQM.root --data --scenario=cosmics --customise DQMServices/Components/test/customHarvesting.py --no_exec --python_filename=test_${tnum}_${step}_3.py

cmsRun -e test_${tnum}_${step}_3.py >& q2.3.log && mv DQM_V0001_R000142560__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000142560__Global__CMSSW_X_Y_Z__RECO_3.root

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_3}.xml

sed -e "s/'file:test_${tnum}_a_1_RAW2DIGI_RECO_DQM_inDQM.root'/'file:test_${tnum}_a_1_RAW2DIGI_RECO_DQM_inDQM.root','file:test_${tnum}_a_2_RAW2DIGI_RECO_DQM_inDQM.root'/" test_${tnum}_${step}_1.py > test_${tnum}_${step}_12.py

cmsRun -e test_${tnum}_${step}_12.py >& q2.12.log && mv DQM_V0001_R000142560__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000142560__Global__CMSSW_X_Y_Z__RECO_12.root

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_12}.xml

sed -e "s/'file:test_${tnum}_a_1_RAW2DIGI_RECO_DQM_inDQM.root'/'file:test_${tnum}_a_1_RAW2DIGI_RECO_DQM_inDQM.root','file:test_${tnum}_a_2_RAW2DIGI_RECO_DQM_inDQM.root','file:test_${tnum}_a_3_RAW2DIGI_RECO_DQM_inDQM.root'/" test_${tnum}_${step}_1.py > test_${tnum}_${step}_123.py

cmsRun -e test_${tnum}_${step}_123.py >& q2.123.log && mv DQM_V0001_R000142560__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000142560__Global__CMSSW_X_Y_Z__RECO_123.root

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_123}.xml
