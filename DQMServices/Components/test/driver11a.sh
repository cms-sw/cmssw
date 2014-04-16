#!/bin/bash

numev=100
tnum=11
DQMSEQUENCE=DQM
step=a

eval `scramv1 r -sh`

cmsDriver.py test_${tnum}_${step}_1 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${numev} --eventcontent DQM --datatier DQMIO --conditions auto:com10  --filein  file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root --data --customise DQMServices/Components/test/customDQM.py --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& p${tnum}.1.log 

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

