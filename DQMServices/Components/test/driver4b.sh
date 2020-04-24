#!/bin/bash

eval `scramv1 r -sh`

tnum=4
step=b
DQMSEQUENCE=HARVESTING:validationHarvestingFS


cmsDriver.py test_${tnum}_${step}_1 -s ${DQMSEQUENCE} --harvesting AtJobEnd --conditions auto:run1_mc --filetype DQM --filein file:test_${tnum}_a_1_inDQM.root --mc --customise DQMServices/Components/test/customHarvesting.py --no_exec --python_filename=test_${tnum}_${tnum}_1.py

cmsRun -e test_${tnum}_${tnum}_1.py >& q${tnum}.1.log && mv DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_4.root

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml
