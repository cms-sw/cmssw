#!/bin/bash

eval `scramv1 r -sh`

tnum=211
numev=1000
step=a
DQMSEQUENCE=DQM:triggerOfflineDQMSource

cmsDriver.py test_${tnum}_${step}_1 -s ${DQMSEQUENCE} -n ${numev} --eventcontent DQM --datatier DQM --conditions auto:com10  --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RECO__PromptReco-v4__165098__4C027417-BA7F-E011-84DA-001617C3B76A.root  --data --customise DQMServices/Components/test/customDQM.py --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& p${tnum}.1.log 

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

