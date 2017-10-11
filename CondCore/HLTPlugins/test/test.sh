#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
# Set SCRAM architecture var
SCRAM_ARCH=slc6_amd64_gcc630;
export SCRAM_ARCH;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;
# Go back to original working directory
cd $W_DIR;
# Run get payload data script

####################
# Test Gains
####################
/afs/cern.ch/user/c/condbpro/public/BROWSER_PI/getPayloadData.py \
    --plugin pluginAlCaRecoTriggerBits_PayloadInspector \
    --plot plot_AlCaRecoTriggerBits_Compare \
    --tag AlCaRecoTriggerBits_TrackerDQM_v2_express \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "284123"}' \
    --db Prod \
    --test;