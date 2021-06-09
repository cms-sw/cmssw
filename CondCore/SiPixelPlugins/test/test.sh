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
    --plugin pluginSiPixel2DTemplateDBObject_PayloadInspector \
    --plot plot_SiPixel2DTemplateHeaderTable \
    --tag SiPixel2DTemplateDBObject_phase2_IT_v6.1.5_25x100_unirradiated_den_v2_mc \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test

mv *.png $W_DIR/results/SiPixelTemplate2DHeader.png

getPayloadData.py \
    --plugin pluginSiPixelFEDChannelContainer_PayloadInspector \
    --plot plot_SiPixelFEDChannelContainerTest \
    --tag SiPixelStatusScenarios_UltraLegacy2018_v0_mc \
    --time_type Run --iovs '{"start_iov": "1", "end_iov" : "1"}' \
    --db Prod \
    --input_params '{"Scenarios":"320824_103,316758_983,320934_254"}' \
    --test ;

mv *.png $W_DIR/results/SiPixelFEDChannelContainer.png


getPayloadData.py \
    --plugin pluginSiPixelFEDChannelContainer_PayloadInspector \
    --plot plot_SiPixelFEDChannelContainerScenarios \
    --tag SiPixelStatusScenarios_UltraLegacy2018_v0_mc \
    --time_type Run --iovs '{"start_iov": "1", "end_iov" : "1"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/results/SiPixelFEDChannelScenarios.png
