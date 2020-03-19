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
if [ -d $W_DIR/plots_LA ]; then
    rm -fr $W_DIR/plots_LA
fi

mkdir $W_DIR/plots_LA

getPayloadData.py \
     --plugin pluginSiPixelLorentzAngle_PayloadInspector \
     --plot plot_SiPixelLorentzAngleValueComparisonTwoTags \
     --tag SiPixelLorentzAngleSim_phase1_BoR3_HV350_Tr1300 \
     --tagtwo SiPixelLorentzAngle_phase1_BoR3_HV350_Tr1300 \
     --time_type Run \
     --iovs '{"start_iov": "1", "end_iov": "1"}' \
     --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
     --db Prod \
     --test ;

mv *.png  $W_DIR/plots_LA/comparisonByValuePhase1.png

getPayloadData.py \
    --plugin pluginSiPixelLorentzAngle_PayloadInspector \
    --plot plot_SiPixelLorentzAngleValues \
    --tag SiPixelLorentzAngle_forWidth_phase1_mc_v1 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test ;

mv *.png  $W_DIR/plots_LA/ByValuePhase1.png

getPayloadData.py \
    --plugin pluginSiPixelLorentzAngle_PayloadInspector \
    --plot plot_SiPixelLorentzAngleValueComparisonTwoTags \
    --tag SiPixelLorentzAngle_forWidth_v1_mc \
    --tagtwo SiPixelLorentzAngle_2016_ultralegacymc_v2 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test ;

mv *.png  $W_DIR/plots_LA/comparisonByValuePhase0.png

getPayloadData.py \
    --plugin pluginSiPixelLorentzAngle_PayloadInspector \
    --plot plot_SiPixelLorentzAngleByRegionComparisonTwoTags \
    --tag SiPixelLorentzAngle_forWidth_v1_mc \
    --tagtwo SiPixelLorentzAngle_2016_ultralegacymc_v2 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test ;

mv *.png  $W_DIR/plots_LA/comparisonByRegionPhase0.png

getPayloadData.py \
     --plugin pluginSiPixelLorentzAngle_PayloadInspector \
     --plot plot_SiPixelBPixLorentzAngleMap \
     --tag SiPixelLorentzAngle_v11_offline \
     --time_type Run \
     --iovs '{"start_iov": "324245", "end_iov": "324245"}' \
     --db Prod \
     --test ;

mv *.png  $W_DIR/plots_LA/SiPixelBPixLorentzAngleMap.png
