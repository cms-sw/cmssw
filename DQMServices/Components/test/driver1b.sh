#!/bin/bash

eval `scramv1 r -sh`

DQMSEQUENCE=HARVESTING:dqmHarvesting
tnum=1
step=b

cmsDriver.py test_${tnum}_${step}_1 -s ${DQMSEQUENCE} --conditions auto:com10 --filein file:test_${tnum}_a_1_RAW2DIGI_RECO_DQM.root --data --customise DQMServices/Components/test/customHarvesting.py --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& q1.1.log && mv DQM_V0001_R000147647__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000147647__Global__CMSSW_X_Y_Z__RECO_1.root

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

cmsDriver.py test_${tnum}_${step}_2 -s ${DQMSEQUENCE} --conditions auto:com10 --filein file:test_${tnum}_a_2_RAW2DIGI_RECO_DQM.root --data --customise DQMServices/Components/test/customHarvesting.py --no_exec --python_filename=test_${tnum}_${step}_2.py

cmsRun -e test_${tnum}_${step}_2.py >& q1.2.log && mv DQM_V0001_R000147647__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000147647__Global__CMSSW_X_Y_Z__RECO_2.root

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_2}.xml


cmsDriver.py test_${tnum}_${step}_3 -s ${DQMSEQUENCE} --conditions auto:com10 --filein file:test_${tnum}_a_3_RAW2DIGI_RECO_DQM.root --data --customise DQMServices/Components/test/customHarvesting.py --no_exec --python_filename=test_${tnum}_${step}_3.py

cmsRun -e test_${tnum}_${step}_3.py >& q1.3.log && mv DQM_V0001_R000147647__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000147647__Global__CMSSW_X_Y_Z__RECO_3.root

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_3}.xml

sed -e "s/'file:test_${tnum}_a_1_RAW2DIGI_RECO_DQM.root'/'file:test_${tnum}_a_1_RAW2DIGI_RECO_DQM.root','file:test_${tnum}_a_2_RAW2DIGI_RECO_DQM.root'/" test_${tnum}_${step}_1.py > test_${tnum}_${step}_12.py

cmsRun -e test_${tnum}_${step}_12.py >& q1.12.log && mv DQM_V0001_R000147647__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000147647__Global__CMSSW_X_Y_Z__RECO_12.root

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_12}.xml


sed -e "s/'file:test_${tnum}_a_1_RAW2DIGI_RECO_DQM.root'/'file:test_${tnum}_a_1_RAW2DIGI_RECO_DQM.root','file:test_${tnum}_a_2_RAW2DIGI_RECO_DQM.root','file:test_${tnum}_a_3_RAW2DIGI_RECO_DQM.root'/" test_${tnum}_${step}_1.py > test_${tnum}_${step}_123.py

cmsRun -e test_${tnum}_${step}_123.py >& q1.123.log && mv DQM_V0001_R000147647__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000147647__Global__CMSSW_X_Y_Z__RECO_123.root

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_123}.xml
