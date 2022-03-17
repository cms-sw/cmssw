#!/bin/bash -ex
#Dataset, Run, Lumi and Events are copied from Workflows 136.8521

if ! edmPickEvents.py "/JetHT/Run2018A-PromptReco-v1/MINIAOD" 315489:31:19015199,315489:31:19098714,315489:31:18897114  > run_edmCopyPickMerge.sh ; then
  cat run_edmCopyPickMerge.sh
  exit 1
fi
chmod +x run_edmCopyPickMerge.sh
./run_edmCopyPickMerge.sh
