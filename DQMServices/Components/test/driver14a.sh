#!/bin/bash

eval `scramv1 r -sh`

DQMSEQUENCE=DQM
numev=100

cmsDriver.py step2_DT1_1 -s RAW2DIGI,RECO,${DQMSEQUENCE} - ${numev} --datatier DQM --eventcontent DQM --conditions auto:com10 --scenario pp  --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root  --data --customise DQMServices/Components/test/customIgProfDQM.py --prefix="igprof -d -t cmsRun -mp -z -o IgProfCumulative_100.gz"

cmsDriver.py step2_DT1_1 -s RAW2DIGI,RECO - ${numev} --datatier RECO --eventcontent RECO --conditions auto:com10 --scenario pp  --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root  --data --customise DQMServices/Components/test/customIgProfRECO.py --prefix="igprof -d -t cmsRun -mp -z -o IgProfRECOCumulative_100.gz"

if [ $? -ne 0 ]; then
	return 1
fi

# igprof-analyse -g -d -v -p -r MEM_TOTAL  -s igprof_RECO_DQM.myrun.gz | sqlite3 igreport_RECO_DQM_MEM_TOTAL.sql3
# igprof-analyse -g -d -v -p -r PERF_TICKS -s igprof_RECO_DQM.myrun.gz | sqlite3 igreport_RECO_DQM_PERF_TICKS.sql3

if [ $? -ne 0 ]; then
	return 1
fi

