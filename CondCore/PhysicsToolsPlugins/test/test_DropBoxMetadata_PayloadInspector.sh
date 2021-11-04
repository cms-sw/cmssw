#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
# Set SCRAM architecture var
SCRAM_ARCH=slc7_amd64_gcc900;
export SCRAM_ARCH;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;

mkdir -p $W_DIR/results

getPayloadData.py \
    --plugin pluginDropBoxMetadata_PayloadInspector \
    --plot plot_DropBoxMetadataTest \
    --tag DropBoxMetadata_v5.1_express \
    --time_type Run \
    --iovs '{"start_iov": "345684", "end_iov": "345684"}' \
    --db Prod \
    --test

getPayloadData.py \
    --plugin pluginDropBoxMetadata_PayloadInspector \
    --plot plot_DropBoxMetadata_Display \
    --tag DropBoxMetadata_v5.1_express \
    --time_type Run \
    --iovs '{"start_iov": "345684", "end_iov": "345684"}' \
    --db Prod \
    --test

getPayloadData.py \
    --plugin pluginDropBoxMetadata_PayloadInspector \
    --plot plot_DropBoxMetadata_Compare \
    --tag DropBoxMetadata_v5.1_express \
    --time_type Run \
    --iovs '{"start_iov": "345684", "end_iov": "346361"}' \
    --db Prod \
    --test
