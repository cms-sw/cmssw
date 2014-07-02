#!/bin/bash

eval `scramv1 r -sh`

DQMSEQUENCE=DQM
tnum=12
step=a

cmsDriver.py test_${tnum}_${step}_1 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n 100 --eventcontent DQM --datatier DQMIO --conditions auto:com10 --scenario HeavyIons  --data --customise DQMServices/Components/test/customHeavyIons.py --processName RECO2 --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& p12.1.log 

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

