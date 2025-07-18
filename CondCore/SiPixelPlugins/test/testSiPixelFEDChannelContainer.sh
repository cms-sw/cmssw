#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;
# Go back to original working directory
cd $W_DIR;
# Run get payload data script
if [ -d $W_DIR/plots_FEDChannelContainer ]; then
    rm -fr $W_DIR/plots_FEDChannelContainer
fi

mkdir $W_DIR/plots_FEDChannelContainer

getPayloadData.py \
    --plugin pluginSiPixelFEDChannelContainer_PayloadInspector \
    --plot plot_SiPixelBPixFEDChannelContainerMap \
    --tag SiPixelStatusScenarios_StuckTBM_2023_v1_mc \
    --input_params '{"Scenarios": "370097_302"}' \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test ;

mv *.png  $W_DIR/plots_FEDChannelContainer/SiPixelBPixFEDChannelContainerMap.png

getPayloadData.py \
    --plugin pluginSiPixelFEDChannelContainer_PayloadInspector \
    --plot plot_SiPixelBPixFEDChannelContainerWeightedMap \
    --tag SiPixelStatusScenarios_StuckTBMandOther_2023_v2_mc \
    --input_params '{"SiPixelQualityProbabilitiesTag": "SiPixelQualityProbabilities_2023_v2_mc"}' \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test ;

mv *.png  $W_DIR/plots_FEDChannelContainer/SiPixelFEDChannelContainerMapWeigthed.png

getPayloadData.py \
    --plugin pluginSiPixelFEDChannelContainer_PayloadInspector \
    --plot plot_SiPixelBPixFEDChannelContainerWeightedMap \
    --tag SiPixelStatusScenarios_StuckTBM_2023_v1_mc \
    --input_params '{"SiPixelQualityProbabilitiesTag": "SiPixelQualityProbabilities_2023_for_eGamma_v1_mc"}' \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test ;

mv *.png  $W_DIR/plots_FEDChannelContainer/SiPixelFEDChannelContainerMapWeigthed_v2.png
