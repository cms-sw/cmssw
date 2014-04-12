#!/bin/bash

eval `scramv1 r -sh`

: ${LOCALRT:?"Need to set CMSSW envs"}

cp $LOCALRT/src/DQMServices/Components/python/test/producePRcfg.py .
python producePRcfg.py
igprof -d -t cmsRun -mp -z -o IgProfCumulative_20.gz cmsRun testPromptReco.py

if [ $? -ne 0 ]; then
  return $?
fi
