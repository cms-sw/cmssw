#!/bin/bash

eval `scramv1 r -sh`

cp ../../producePRcfg.py .
python producePRcfg.py
igprof -d -t cmsRunGlibC -mp -z -o IgProfCumulative_100.gz cmsRunGlibC testPromptReco.py

if [ $? -ne 0 ]; then
  return 1
fi

# igprof-analyse -g -d -v -p -r MEM_TOTAL  -s igprof_RECO_DQM.myrun.gz | sqlite3 igreport_RECO_DQM_MEM_TOTAL.sql3
# igprof-analyse -g -d -v -p -r PERF_TICKS -s igprof_RECO_DQM.myrun.gz | sqlite3 igreport_RECO_DQM_PERF_TICKS.sql3


