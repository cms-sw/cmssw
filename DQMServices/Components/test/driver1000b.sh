#!/bin/bash

eval `scramv1 r -sh`

TNUM=1000
DQMSEQUENCE=HARVESTING:validationHarvesting+dqmHarvesting
NUMEV=1
STEP=b

cmsDriver.py test_${TNUM}_${STEP}_1 --no_exec --scenario HeavyIons --conditions auto:starthi_HIon --mc -s ${DQMSEQUENCE} -n ${NUMEV} --filein file:test_${TNUM}_a_1.root --customise DQMServices/Components/test/customHarvesting.py --python_filename=test_${TNUM}_${STEP}_1.py


cmsRun -e test_${TNUM}_${STEP}_1.py >& q${TNUM}.1.log && mv DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_1000.root


if [ $? -ne 0 ]; then
	exit 1
fi

mv FrameworkJobReport{,_${TNUM}_${STEP}_1}.xml
