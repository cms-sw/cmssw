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

if [ -d $W_DIR/plots ]; then
    rm -fr $W_DIR/plots
fi

mkdir $W_DIR/plots

####################
# Test Gains
####################
getPayloadData.py \
    --plugin pluginSiStripApvGain_PayloadInspector \
    --plot plot_SiStripApvGainsValuesComparator \
    --tag SiStripApvGainAfterAbortGap_PCL_v0_prompt \
    --time_type Run \
    --iovs '{"start_iov": "302393", "end_iov": "305114"}' \
    --db Prep \
    --test;

mv *.png $W_DIR/plots/G2_Value_update.png

getPayloadData.py \
    --plugin pluginSiStripApvGain_PayloadInspector \
    --plot plot_SiStripApvGainsMaxDeviationRatio2sigmaTrackerMap \
    --tag SiStripApvGainAfterAbortGap_PCL_v0_prompt \
    --time_type Run \
    --iovs '{"start_iov": "302393", "end_iov": "305114"}' \
    --db Prep \
    --test;

mv *.png $W_DIR/plots/G2_MaxDeviatonRatio_update.png

getPayloadData.py \
    --plugin pluginSiStripApvGain_PayloadInspector \
    --plot plot_SiStripApvGainsRatioComparatorByRegion \
    --tag SiStripApvGainAfterAbortGap_PCL_v0_prompt \
    --time_type Run \
    --iovs '{"start_iov": "280000", "end_iov": "305114"}' \
    --db Prep \
    --test;

mv *.png $W_DIR/plots/G2_Ratio_update.png

getPayloadData.py \
    --plugin pluginSiStripApvGain_PayloadInspector \
    --plot plot_SiStripApvGainsValuesComparator \
    --tag SiStripApvGain_GR10_v1_hlt \
    --time_type Run \
    --iovs '{"start_iov": "302322", "end_iov": "306054"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/plots/G1_Value_update.png