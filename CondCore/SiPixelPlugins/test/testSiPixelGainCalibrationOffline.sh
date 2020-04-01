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
if [ -d $W_DIR/plots_GainCalib ]; then
    rm -fr $W_DIR/plots_GainCalib
fi

mkdir $W_DIR/plots_GainCalib

## single IOV plots

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibrationOfflineGainsValues \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalib/SingleIOVGains.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibrationOfflinePedestalsValues \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalib/SingleIOVPedestals.png

## single IOV by Partitions

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibrationOfflineGainsByPart \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalib/SingleIOVGainsByPart.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibrationOfflinePedestalsByPart \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalib/SingleIOVPedestalsByPart.png

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

mv *.png  $W_DIR/plots_GainCalib/TwoTagsGainsComparison.png

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

mv *.png  $W_DIR/plots_GainCalib/TwoTagsPedestalsComparison.png

## single tag, two IOVs plots

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflineGainComparisonSingleTag \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "326851"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalib/SingleTagGainsComparison.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflinePedestalComparisonSingleTag \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "326851"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalib/SingleTagPedestalsComparison.png

## Maps

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflinePedestalsBPIXMap \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalib/PedestalBPixMap.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflineGainsBPIXMap \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalib/GainBPixMap.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflinePedestalsFPIXMap \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalib/PedestalFPixMap.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflineGainsFPIXMap \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalib/GainFPixMap.png

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

mv *.png  $W_DIR/plots_GainCalib/TwoTagsGainsByRegionComparison.png

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

mv *.png  $W_DIR/plots_GainCalib/TwoTagsPedestalsByRegionComparison.png

## single tag, two IOVs plots by Region

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflineGainByRegionComparisonSingleTag \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "326851"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalib/SingleTagGainsByRegionComparison.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibOfflinePedestalByRegionComparisonSingleTag \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "326851"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalib/SingleTagPedestalsByRegionComparison.png

## correlations

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationOffline_PayloadInspector \
    --plot plot_SiPixelGainCalibrationCorrelations \
    --tag SiPixelGainCalibration_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test ;

mv *.png  $W_DIR/plots_GainCalib/SingleTagGainsPedestalsCorrelations.png
