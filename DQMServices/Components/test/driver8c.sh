#!/bin/bash

eval `scramv1 r -sh`

step=c
tnum=8
numev=10
DQMSEQUENCE=HARVESTING:validationHarvesting+dqmHarvesting


cmsDriver.py step3  --conditions auto:startup -s ${DQMSEQUENCE} --datatier DQM --eventcontent DQM -n ${numev} --python_filename=test_${tnum}_${step}_1.py --filein file:test_${tnum}_b_1.root --customise DQMServices/Components/test/customHarvesting.py


cmsRun -e test_${tnum}_${step}_1.py >& p${tnum}.${step}.1.log

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml
