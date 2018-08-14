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
getPayloadData.py \
    --plugin pluginAlignPCLThresholds_PayloadInspector \
    --plot plot_AlignPCLThresholds_Display \
    --tag SiPixelAliThresholds_offline_v0 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;