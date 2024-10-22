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
if [ -d $W_DIR/plots_GainCalibForHLT ]; then
    rm -fr $W_DIR/plots_GainCalibForHLT
fi

mkdir $W_DIR/plots_GainCalibForHLT

## single IOV plots

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibrationForHLTGainsValues \
    --tag SiPixelGainCalibrationHLT_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/SingleIOVGains.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibrationForHLTPedestalsValues \
    --tag SiPixelGainCalibrationHLT_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/SingleIOVPedestals.png

## single IOV by Partitions

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibrationForHLTGainsByPart \
    --tag SiPixelGainCalibrationHLT_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/SingleIOVGainsByPart.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibrationForHLTPedestalsByPart \
    --tag SiPixelGainCalibrationHLT_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/SingleIOVPedestalsByPart.png

## two tags plots

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibForHLTGainComparisonTwoTags \
    --tag SiPixelGainCalibration_hlt_phase1_mc_v2 \
    --tagtwo SiPixelGainCalibration_hlt_phase1_mc_v3 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/TwoTagsGainsComparison.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibForHLTPedestalComparisonTwoTags \
    --tag SiPixelGainCalibration_hlt_phase1_mc_v2 \
    --tagtwo SiPixelGainCalibration_hlt_phase1_mc_v3 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/TwoTagsPedestalsComparison.png

## single tag, two IOVs plots

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibForHLTGainComparisonSingleTag \
    --tag SiPixelGainCalibrationHLT_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "326851"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/SingleTagGainsComparison.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibForHLTPedestalComparisonSingleTag \
    --tag SiPixelGainCalibrationHLT_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "326851"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/SingleTagPedestalsComparison.png

## Maps

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibForHLTPedestalsBPIXMap \
    --tag SiPixelGainCalibrationHLT_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/PedestalBPixMap.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibForHLTGainsBPIXMap \
    --tag SiPixelGainCalibrationHLT_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/GainBPixMap.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibForHLTPedestalsFPIXMap \
    --tag SiPixelGainCalibrationHLT_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/PedestalFPixMap.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibForHLTGainsFPIXMap \
    --tag SiPixelGainCalibrationHLT_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/GainFPixMap.png

## two tags plots by Region

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibForHLTGainByRegionComparisonTwoTags \
    --tag SiPixelGainCalibration_hlt_phase1_mc_v2 \
    --tagtwo SiPixelGainCalibration_hlt_phase1_mc_v3 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/TwoTagsGainsByRegionComparison.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibForHLTPedestalByRegionComparisonTwoTags \
    --tag SiPixelGainCalibration_hlt_phase1_mc_v2 \
    --tagtwo SiPixelGainCalibration_hlt_phase1_mc_v3 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/TwoTagsPedestalsByRegionComparison.png

## single tag, two IOVs plots by Region

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibForHLTGainByRegionComparisonSingleTag \
    --tag SiPixelGainCalibrationHLT_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "326851"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/SingleTagGainsByRegionComparison.png

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibForHLTPedestalByRegionComparisonSingleTag \
    --tag SiPixelGainCalibrationHLT_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "326851"}' \
    --db Prod \
    --test;

mv *.png  $W_DIR/plots_GainCalibForHLT/SingleTagPedestalsByRegionComparison.png

## correlations

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibrationForHLTCorrelations \
    --tag SiPixelGainCalibrationHLT_2009runs_express \
    --time_type Run \
    --iovs '{"start_iov": "312203", "end_iov": "312203"}' \
    --db Prod \
    --test ;

mv *.png  $W_DIR/plots_GainCalibForHLT/SingleTagGainsPedestalsCorrelations.png

## diff and ratio

getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibForHLTGainDiffRatioTwoTags \
    --tagtwo SiPixelGainCalibrationHLT_2009runs_hlt \
    --tag SiPixelGainCalibrationHLT_2009runs_ScaledForVCal_hlt \
    --time_type Run \
    --iovs '{"start_iov": "310000", "end_iov": "310000"}' \
    --iovstwo '{"start_iov": "310000", "end_iov": "310000"}'  \
    --db Prod \
    --test ;

mv *.png  $W_DIR/plots_GainCalibForHLT/DiffAndRatio.png

## diff and ratio reverse
getPayloadData.py \
    --plugin pluginSiPixelGainCalibrationForHLT_PayloadInspector \
    --plot plot_SiPixelGainCalibForHLTGainDiffRatioTwoTags \
    --tagtwo SiPixelGainCalibrationHLT_2009runs_ScaledForVCal_hlt \
    --tag SiPixelGainCalibrationHLT_2009runs_hlt \
    --time_type Run \
    --iovs '{"start_iov": "310000", "end_iov": "310000"}' \
    --iovstwo '{"start_iov": "310000", "end_iov": "310000"}' \
    --db Prod \
    --test ;

mv *.png  $W_DIR/plots_GainCalibForHLT/DiffAndRatio_reverse.png
