#!/bin/bash

eval `scramv1 r -sh`

tnum=2
numev=500
step=a
DQMSEQUENCE=DQM

cmsDriver.py test_${tnum}_${step}_1 -s RAW2DIGI,RECO:reconstructionCosmics,${DQMSEQUENCE} -n ${numev} --eventcontent RECO,DQM --datatier RECO,DQMIO --conditions auto:com10  --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/Cosmics__RAW__v1__142560__0084A7EF-8CA3-DF11-B661-0030487C5CE2.root --data --scenario cosmics --customise DQMServices/Components/test/customReco.py --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& p2.1.log 

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

cmsDriver.py test_${tnum}_${step}_2 -s RAW2DIGI,RECO:reconstructionCosmics,${DQMSEQUENCE} -n ${numev} --eventcontent RECO,DQM --datatier RECO,DQMIO --conditions auto:com10  --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/Cosmics__RAW__v1__142560__0089307A-6AA3-DF11-806D-001617E30D40.root --data --scenario cosmics --customise DQMServices/Components/test/customReco.py --no_exec --python_filename=test_${tnum}_${step}_2.py

cmsRun -e test_${tnum}_${step}_2.py >& p2.2.log 

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_2}.xml

cmsDriver.py test_${tnum}_${step}_3 -s RAW2DIGI,RECO:reconstructionCosmics,${DQMSEQUENCE} -n ${numev} --eventcontent RECO,DQM --datatier RECO,DQMIO --conditions auto:com10  --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/Cosmics__RAW__v1__142560__026275A7-81A3-DF11-BDEE-001617C3B5D8.root --data --scenario cosmics --customise DQMServices/Components/test/customReco.py --no_exec --python_filename=test_${tnum}_${step}_3.py

cmsRun -e test_${tnum}_${step}_3.py >& p2.3.log 

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_3}.xml
