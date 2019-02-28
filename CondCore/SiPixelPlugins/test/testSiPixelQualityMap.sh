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

#    --iovs '{"start_iov": "1407898869563565", "end_iov": "1407898869563565"}' \
####################
# Test SiPixelQuality
####################
getPayloadData.py \
    --plugin pluginSiPixelQuality_PayloadInspector \
    --plot plot_SiPixelBPixQualityMap \
    --tag  SiPixelQuality_byPCL_prompt_v2 \
    --time_type Lumi \
    --iovs '{"start_iov": "1390517136916505", "end_iov": "1390517136916505"}' \
    --db Prod \
    --test;

getPayloadData.py \
    --plugin pluginSiPixelQuality_PayloadInspector \
    --plot plot_SiPixelFPixQualityMap \
    --tag  SiPixelQuality_byPCL_prompt_v2 \
    --time_type Lumi \
    --iovs '{"start_iov": "1390517136916505", "end_iov": "1390517136916505"}' \
    --db Prod \
    --test;