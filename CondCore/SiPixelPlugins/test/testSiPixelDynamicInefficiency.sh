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

if [ -d $W_DIR/plots_DynIneff ]; then
    rm -fr $W_DIR/plots_DynIneff
fi

mkdir $W_DIR/plots_DynIneff

# Run get payload data script
getPayloadData.py \
    --plugin pluginSiPixelDynamicInefficiency_PayloadInspector \
    --plot plot_SiPixelDynamicInefficiencyTest \
    --tag SiPixelDynamicInefficiency_PhaseI_v9 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

getPayloadData.py \
    --plugin pluginSiPixelDynamicInefficiency_PayloadInspector \
    --plot plot_SiPixelBPixIneffROCfromDynIneffMap \
    --tag SiPixelDynamicInefficiency_PhaseI_v9 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/plots_DynIneff/SiPixelBPixIneffROCfromDynIneffMap.png

getPayloadData.py \
    --plugin pluginSiPixelDynamicInefficiency_PayloadInspector \
    --plot plot_SiPixelFPixIneffROCfromDynIneffMap \
    --tag SiPixelDynamicInefficiency_PhaseI_v9 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/plots_DynIneff/SiPixelFPixIneffROCfromDynIneffMap.png

getPayloadData.py \
    --plugin pluginSiPixelDynamicInefficiency_PayloadInspector \
    --plot plot_SiPixelFullIneffROCfromDynIneffMap \
    --tag SiPixelDynamicInefficiency_PhaseI_v9 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/plots_DynIneff/SiPixelFullIneffROCfromDynIneffMap.png

getPayloadData.py \
    --plugin pluginSiPixelDynamicInefficiency_PayloadInspector \
    --plot plot_SiPixelFullIneffROCsMapCompareTwoTags \
    --tag SiPixelDynamicInefficiency_PhaseI_Run3Studies_v2 \
    --tagtwo SiPixelDynamicInefficiency_PhaseI_v9 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/plots_DynIneff/SiPixelFullIneffROCfromDynIneffMapDelta.png
