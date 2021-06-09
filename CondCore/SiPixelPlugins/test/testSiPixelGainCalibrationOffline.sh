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
if [ -d $W_DIR/plots_GainCalibOffline ]; then
    rm -fr $W_DIR/plots_GainCalibOffline
fi

mkdir $W_DIR/plots_GainCalibOffline

## single IOV plots

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibrationOfflineGainsValues \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/SingleIOVGains.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibrationOfflinePedestalsValues \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/SingleIOVPedestals.png

## single IOV by Partitions

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibrationOfflineGainsByPart \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/SingleIOVGainsByPart.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibrationOfflinePedestalsByPart \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/SingleIOVPedestalsByPart.png

## two tags plots

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflineGainComparisonTwoTags \
    --tag SiPixelGainCalibration_phase1_mc_v2 \
    --tagtwo SiPixelGainCalibration_phase1_mc_v3 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/TwoTagsGainsComparison.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflinePedestalComparisonTwoTags \
    --tag SiPixelGainCalibration_phase1_mc_v2 \
    --tagtwo SiPixelGainCalibration_phase1_mc_v3 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/TwoTagsPedestalsComparison.png

## single tag, two IOVs plots

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflineGainComparisonSingleTag \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "326851"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/SingleTagGainsComparison.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflinePedestalComparisonSingleTag \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "326851"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/SingleTagPedestalsComparison.png

## Maps

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflinePedestalsBPIXMap \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/PedestalBPixMap.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflineGainsBPIXMap \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/GainBPixMap.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflinePedestalsFPIXMap \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/PedestalFPixMap.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflineGainsFPIXMap \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/GainFPixMap.png

## two tags plots by Region

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflineGainByRegionComparisonTwoTags \
    --tag SiPixelGainCalibration_phase1_mc_v2 \
    --tagtwo SiPixelGainCalibration_phase1_mc_v3 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/TwoTagsGainsByRegionComparison.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflinePedestalByRegionComparisonTwoTags \
    --tag SiPixelGainCalibration_phase1_mc_v2 \
    --tagtwo SiPixelGainCalibration_phase1_mc_v3 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/TwoTagsPedestalsByRegionComparison.png

## single tag, two IOVs plots by Region

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflineGainByRegionComparisonSingleTag \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "326851"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/SingleTagGainsByRegionComparison.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflinePedestalByRegionComparisonSingleTag \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "326851"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibOffline/SingleTagPedestalsByRegionComparison.png

## correlations

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibrationOfflineCorrelations \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test ;

mv *.png  $W_DIR/plots_GainCalibOffline/SingleTagGainsPedestalsCorrelations.png
