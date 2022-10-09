#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;
# Go back to original working directory
cd $W_DIR;
# Run get payload data script

mkdir -p $W_DIR/results

if [ -f *.png ]; then
    rm *.png
fi

####################
# Test Display
####################
getPayloadData.py \
    --plugin pluginAlignPCLThresholds_PayloadInspector \
    --plot plot_AlignPCLThresholds_Display \
    --tag SiPixelAliThresholds_offline_v0 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/LG_display.png

####################
# Test Compare
####################
getPayloadData.py \
    --plugin pluginAlignPCLThresholds_PayloadInspector \
    --plot plot_AlignPCLThresholds_CompareTwoTags \
    --tag SiPixelAliThresholds_offline_v0 \
    --tagtwo SiPixelAliThresholds_express_v0 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/LG_compare.png

####################
# Test Display HG
####################
getPayloadData.py \
    --plugin pluginAlignPCLThresholdsHG_PayloadInspector \
    --plot plot_AlignPCLThresholdsHG_Display \
    --tag SiPixelAliThresholdsHG_express_v0 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/HG_display.png

getPayloadData.py \
    --plugin pluginAlignPCLThresholdsHG_PayloadInspector \
    --plot plot_AlignPCLThresholdsHG_Display \
    --tag SiPixelAliThresholdsHG_express_v0 \
    --time_type Run \
    --iovs '{"start_iov": "359659", "end_iov": "359659"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/HG_display_IOV2.png

####################
# Test Compare HG
####################
getPayloadData.py \
    --plugin pluginAlignPCLThresholdsHG_PayloadInspector \
    --plot plot_AlignPCLThresholdsHG_Compare \
    --tag SiPixelAliThresholdsHG_express_v0 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "359659"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/HG_compare.png
