#!/bin/bash

eval `scramv1 r -sh`

tnum=11
DQMSEQUENCE=HARVESTING:dqmHarvesting
step=b

#cmsDriver.py test_${tnum}_${step}_1 -s ${DQMSEQUENCE} --conditions auto:com10 --data --filetype DQM --filein file:test_${tnum}_a_1_RAW2DIGI_RECO_DQM.root --scenario pp --customise DQMServices/Components/test/customHarvesting.py --no_exec --python_filename=test_${tnum}_${step}_1.py
cmsDriver.py test_${tnum}_${step}_1 -s ${DQMSEQUENCE} --conditions 81X_dataRun2_v9 --data --filetype DQM --filein file:test_${tnum}_a_1_RAW2DIGI_RECO_DQM.root --scenario pp --customise DQMServices/Components/test/customHarvesting.py --no_exec --python_filename=test_${tnum}_${step}_1.py --era Run2_2016
#cmsRun -e test_${tnum}_${step}_1.py 2>&1 | tee q${tnum}.1.log 

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

