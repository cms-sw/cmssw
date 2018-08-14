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

mkdir -p $W_DIR/plots

####################
# Test display 
####################
getPayloadData.py \
    --plugin pluginAlCaRecoTriggerBits_PayloadInspector \
    --plot plot_AlCaRecoTriggerBits_Display \
    --tag AlCaRecoHLTpaths8e29_1e31_v7_hlt \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/plots/AlCaRecoTriggerBits_Display.png

####################
# Test compare
####################
getPayloadData.py \
    --plugin pluginAlCaRecoTriggerBits_PayloadInspector \
    --plot plot_AlCaRecoTriggerBits_Compare \
    --tag AlCaRecoHLTpaths8e29_1e31_v7_hlt \
    --time_type Run \
    --iovs '{"start_iov": "270000", "end_iov": "304820"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/plots/AlCaRecoTriggerBits_Compare.png