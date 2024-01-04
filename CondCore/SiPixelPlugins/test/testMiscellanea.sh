#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
# Set SCRAM architecture var
SCRAM_ARCH=slc7_amd64_gcc900;
export SCRAM_ARCH;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;

mkdir -p $W_DIR/display

getPayloadData.py \
    --plugin pluginSiPixelTemplateDBObject_PayloadInspector \
    --plot plot_SiPixelTemplateIDsFullPixelMap \
    --tag SiPixelTemplateDBObject_phase1_38T_2018_ultralegacymc_v1 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov" : "1"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/display/testPixelMap.png

# test a given PU bin
getPayloadData.py \
    --plugin pluginSiPixelQualityProbabilities_PayloadInspector \
    --plot plot_SiPixelQualityProbabilityDensityPerPUbin \
    --tag SiPixelQualityProbabilities_UltraLegacy2018_v0_mc \
    --input_params '{"PU bin": "10"}' \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/display/testSiPixelQualityProbabilityDensity.png

# test all PU bins
getPayloadData.py \
    --plugin pluginSiPixelQualityProbabilities_PayloadInspector \
    --plot plot_SiPixelQualityProbabilityDensityPerPUbin \
    --tag SiPixelQualityProbabilities_2023_v2_BPix_mc \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test ;

mv *.png $W_DIR/display/testSiPixelQualityProbabilityDensity_v2.png
