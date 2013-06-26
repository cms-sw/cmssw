#!/bin/bash

eval `scramv1 r -sh`

tnum=9
step=b
DQMSEQUENCE=HARVESTING:validationHarvestingFS

cmsDriver.py test_${tnum}_${step}_1 -s ${DQMSEQUENCE} --no_exec --harvesting AtRunEnd --conditions auto:startup --filein file:test_${tnum}_a_1.root --python_filename=test_${tnum}_${step}_1.py --mc --customise DQMServices/Components/test/customHarvesting.py

cmsRun -e test_${tnum}_${step}_1.py >& p${tnum}.${step}.1.log 

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

