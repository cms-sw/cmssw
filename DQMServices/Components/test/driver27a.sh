#!/bin/bash

numev=500
tnum=27
DQMSEQUENCE=DQM
step=a

eval `scramv1 r -sh`

cmsDriver.py test_${tnum}_${step}_1 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${numev} --eventcontent DQM --conditions auto:com10  --filein  /store/data/Run2012B/VBF1Parked/RAW/v1/000/194/315/00A214FF-FD9F-E111-88F1-5404A638869B.root --data --customise DQMServices/Components/test/customDQM.py --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& p${tnum}.1.log 

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

