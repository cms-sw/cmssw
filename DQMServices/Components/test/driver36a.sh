#!/bin/bash

eval `scramv1 r -sh`

tnum=36
numev=100
DQMSEQUENCE=DQM
FILENAME="file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MultiJet__RAW__v1__180252__E85462BF-1A03-E111-84FA-BCAEC5364C42.root"

cmsDriver.py SingleMonitoring -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${numev} --eventcontent DQM --filein ${FILENAME} --conditions auto:com10 --data --no_exec --python_filename=test_${tnum}.py --customise DQMServices/Components/test/customIgProfDQM.py

igprof -d -t cmsRun -pp -z -o IgProfCumulative_${numev}.gz cmsRun test_${tnum}.py &> p${tnum}.1.log

if [ $? -ne 0 ]; then
  return $?
fi

