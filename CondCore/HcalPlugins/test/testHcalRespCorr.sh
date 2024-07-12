#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;
# Go back to original working directory
cd $W_DIR;
# Run get payload data script

if [ -d $W_DIR/plots ]; then
    rm -fr $W_DIR/plots
fi

mkdir $W_DIR/plots

getPayloadData.py \
    --plugin pluginHcalRespCorrs_PayloadInspector \
    --plot plot_HcalRespCorrsRatioAll \
    --tag HcalRespCorrs_2024_v1.0_data \
    --tagtwo HcalRespCorrs_2024_v2.0_data  \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/plots/HcalRespCorrsRatio2D.png

getPayloadData.py \
    --plugin pluginHcalRespCorrs_PayloadInspector \
    --plot plot_HcalRespCorrsComparatorTwoTags \
    --tag HcalRespCorrs_2024_v1.0_data \
    --tagtwo HcalRespCorrs_2024_v2.0_data  \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/plots/HcalRespCorrsCompareTwoTags.png

getPayloadData.py \
    --plugin pluginHcalRespCorrs_PayloadInspector \
    --plot plot_HcalRespCorrsCorrelationTwoTags \
    --tag HcalRespCorrs_2024_v1.0_data \
    --tagtwo HcalRespCorrs_2024_v2.0_data  \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/plots/HcalRespCorrsCorrelateTwoTags.png
