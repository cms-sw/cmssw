#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
# Set SCRAM architecture var
SCRAM_ARCH=slc7_amd64_gcc900;
export SCRAM_ARCH;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;
# Go back to original working directory
cd $W_DIR;
# Run get payload data script

####################
# Test SiPixelQuality
####################
# getPayloadData.py \
#     --plugin pluginSiPixelQuality_PayloadInspector \
#     --plot plot_SiPixelQualityTest \
#     --tag SiPixelQuality_byPCL_prompt_v2 \
#     --time_type Run \
#     --iovs '{"start_iov": "1395142816694363", "end_iov": "1395142816694363"}' \
#     --db Prod \
#     --test;

# getPayloadData.py \
#     --plugin pluginSiPixelQuality_PayloadInspector \
#     --plot plot_SiPixelQualityBadRocsTimeHistory \
#     --tag SiPixelQuality_byPCL_prompt_v2 \
#     --time_type Lumi \
#     --iovs '{"start_iov": "1390615921164381", "end_iov": "1395142816694363"}' \
#     --db Prod \
#     --test;

#    --tag SiPixelQuality_byPCL_stuckTBM_v1 \
#    --tag SiPixelQuality_byPCL_other_v1 \
#    --tag SiPixelQuality_byPCL_prompt_v2 \

### to produce detailed list of bad ROCs for a whole year

getPayloadData.py \
    --plugin pluginSiPixelQuality_PayloadInspector \
    --plot plot_SiPixelQualityBadRocsSummary \
    --tag SiPixelQuality_byPCL_stuckTBM_v1 \
    --time_type Lumi \
    --iovs '{"start_iov": "1355878225674241", "end_iov": "1395228716040193"}' \
    --db Prod \
    --test;

