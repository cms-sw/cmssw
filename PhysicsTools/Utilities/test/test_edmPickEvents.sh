#!/bin/bash -ex
#Dataset, Run, Lumi and Events are copied from Workflows 4.22

edmPickEvents.py  "/Cosmics/Run2011A-v1/RAW" 160960:277:10001082,160960:277:10001058,160960:277:10001650 > run_edmCopyPickMerge.sh
chmod +x run_edmCopyPickMerge.sh
./run_edmCopyPickMerge.sh
