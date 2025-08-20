#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
# Set SCRAM architecture var
SCRAM_ARCH=slc7_amd64_gcc900;
export SCRAM_ARCH;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;

mkdir -p $W_DIR/plots

getPayloadData.py \
    --plugin pluginDeDxCalibration_PayloadInspector \
    --plot plot_DeDxCalibrationTest \
    --tag DeDxCalibration_HI_2024_v1 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test

getPayloadData.py \
    --plugin pluginDeDxCalibration_PayloadInspector \
    --plot plot_DeDxCalibrationInspector \
    --tag DeDxCalibration_HI_2024_v2 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test

mv *.png $W_DIR/plots/Inspector.png

getPayloadData.py \
    --plugin pluginDeDxCalibration_PayloadInspector \
    --plot plot_DeDxCalibrationPlot \
    --tag DeDxCalibration_HI_2024_v2 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test

mv *.png $W_DIR/plots/Plot.png
