#!/bin/bash

eval `scramv1 r -sh`

DQMSEQUENCE=HARVESTING:dqmHarvesting

function doReport {
    igprof-analyse -g -d -v -p -r $1   -s IgProf.$2.gz   | sqlite3 igreport_$2_$1.sql3
    igprof-analyse -g -d -v -p -r $1   -s IgProfRECO.$2.gz   | sqlite3 igreportRECO_$2_$1.sql3
}

function doCumulative {
    igprof-analyse -g -d -v -p -r $1   -s IgProf$2.gz   | sqlite3 igreport_$2_$1.sql3
    igprof-analyse -g -d -v -p -r $1   -s IgProfRECO$2.gz   | sqlite3 igreportRECO_$2_$1.sql3
}

doReport MEM_LIVE 1
doReport MEM_LIVE 26
doReport MEM_LIVE 51
doReport MEM_LIVE 76
doReport MEM_TOTAL 1
doReport MEM_TOTAL 26
doReport MEM_TOTAL 51
doReport MEM_TOTAL 76
doCumulative MEM_LIVE Cumulative_100
doCumulative MEM_TOT Cumulative_100

cmsDriver.py step3_DT1_1 -s ${DQMSEQUENCE} --conditions auto:com10 --data --filein file:step2_DT1_1_RAW2DIGI_RECO_DQM.root --scenario pp --customise DQMServices/Components/test/customHarvesting.py --no_exec

(time igprof -d -t cmsRun -mp -z -o igprof_HARVESTING.myrun.gz cmsRun step3_DT1_1_HARVESTING.py) >& out_HARVESTING.myrun.txt </dev/null 

if [ $? -ne 0 ]; then
  return 1
fi

igprof-analyse -g -d -v -p -r MEM_LIVE   -s igprof_HARVESTING.myrun.gz | sqlite3 igreport_RECO_DQM_MEM_LIVE.sql3
igprof-analyse -g -d -v -p -r MEM_TOTAL  -s igprof_HARVESTING.myrun.gz | sqlite3 igreport_RECO_DQM_MEM_TOTAL.sql3
igprof-analyse -g -d -v -p -r PERF_TICKS -s igprof_HARVESTING.myrun.gz | sqlite3 igreport_RECO_DQM_PERF_TICKS.sql3
