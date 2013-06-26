#!/bin/bash

eval `scramv1 r -sh`

TNUM=10
DQMSEQUENCE=DQM
NUMEV=500

cmsDriver.py step2_DT1_1 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${NUMEV} --eventcontent RECO --conditions auto:craft09  --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/Cosmics__RAW__v1__110998__001404E1-0F8A-DE11-ADB3-000423D99EEE.root --data  --prefix "time valgrind --tool=memcheck `cmsvgsupp` --num-callers=20 --xml=yes --xml-file=step2_DT1_valgrind_memcheck.xml " --customise DQMServices/Components/test/customReco.py --no_exec --python_filename=step${TNUM}_1.py

cmsRun -e step${TNUM}_1.py >& p1.1.log 

if [ $? -ne 0 ]; then
	exit 1
fi

cmsDriver.py step2_DT1_2 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${NUMEV} --eventcontent RECO --conditions auto:craft09  --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/Cosmics__RAW__v1__110998__002174A8-E989-DE11-8B4D-000423D6CA42.root --data --customise DQMServices/Components/test/customReco.py --no_exec --python_filename=step${TNUM}_2.py

cmsRun -e step${TNUM}_2.py >& p1.2.log 

if [ $? -ne 0 ]; then
	exit 1
fi

cmsDriver.py step2_DT1_3 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${NUMEV} --eventcontent RECO --conditions auto:craft09  --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/Cosmics__RAW__v1__110998__0053DAF7-E889-DE11-B3D7-000423D98834.root --data --customise DQMServices/Components/test/customReco.py --no_exec --python_filename=step${TNUM}_3.py

cmsRun -e step${TNUM}_3.py >& p1.3.log 


if [ $? -ne 0 ]; then
	exit 1
fi
