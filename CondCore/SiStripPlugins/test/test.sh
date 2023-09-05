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

mkdir -p $W_DIR/results

if [ -f *.png ]; then
    rm *.png
fi

####################
# Test Gains
####################
getPayloadData.py \
    --plugin pluginSiStripApvGain_PayloadInspector \
    --plot plot_SiStripApvGainsByRegion \
    --tag SiStripApvGain_FromParticles_GR10_v1_express \
    --time_type Run \
    --iovs '{"start_iov": "286042", "end_iov": "286042"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/SiStripApvGainsByRegion.png

######################
# Test Lorentz Angle
######################
getPayloadData.py \
    --plugin pluginSiStripLorentzAngle_PayloadInspector \
    --plot plot_SiStripLorentzAngleByRegion \
    --tag  SiStripLorentzAngleDeco_GR10_v1_prompt \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/SiStripLorentzAngleByRegion.png

######################
# Test Lorentz Angle Comparison
######################
getPayloadData.py \
    --plugin pluginSiStripLorentzAngle_PayloadInspector \
    --plot plot_SiStripLorentzAngleByRegionCompareSingleTag \
    --tag SiStripLorentzAngleDeco_GR10_v1_prompt \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "400000"}' \
    --db Prod --test ;

######################
# Test Backplane correction
######################
getPayloadData.py \
    --plugin pluginSiStripBackPlaneCorrection_PayloadInspector \
    --plot plot_SiStripBackPlaneCorrectionByRegion \
    --tag SiStripBackPlaneCorrection_deco_GR10_v1_express \
    --time_type Run \
    --iovs '{"start_iov": "153690", "end_iov": "153690"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/SiStripBackPlaneCorrectionByRegion.png

######################
# Test Bad components
######################
getPayloadData.py \
    --plugin pluginSiStripBadStrip_PayloadInspector \
    --plot plot_SiStripBadStripQualityAnalysis \
    --tag  SiStripBadComponents_startupMC_for2017_v1_mc\
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/SiStripBadStripQualityAnalysis.png

######################
# Test Conf Object
######################
getPayloadData.py \
    --plugin pluginSiStripConfObject_PayloadInspector \
    --plot plot_SiStripConfObjectDisplay \
    --tag SiStripShiftAndCrosstalk_GR10_v1_express \
    --time_type Run --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/SiStripConfObjectDisplay.png

######################
# Test DetVOff
######################
getPayloadData.py \
    --plugin pluginSiStripDetVOff_PayloadInspector \
    --plot plot_SiStripDetVOffTest \
    --tag SiStripDetVOff_v3_offline \
    --time_type Time \
    --iovs '{"start_iov": "685006631803433472", "end_iov": "6850066318803433472"}' \
    --db Prod \
    --test ;

getPayloadData.py \
    --plugin pluginSiStripDetVOff_PayloadInspector \
    --plot plot_SiStripDetVOffByRegion \
    --tag SiStripDetVOff_v6_prompt \
    --time_type Run --iovs '{"start_iov": "6607932533539533824", "end_iov": "6607932533539533824"}' \
    --db Prod \
    --test;

######################
# Test dumping of switched off modules
######################
getPayloadData.py \
    --plugin pluginSiStripDetVOff_PayloadInspector \
    --plot plot_SiStripLVOffListOfModules \
    --tag SiStripDetVOff_v3_offline \
    --time_type Time \
    --iovs '{"start_iov": "6850066318803433472", "end_iov": "6850066318803433472"}' \
    --db Prod \
    --test;

######################
# Test SiStripBadStripFractionTH2PolyTkMap
######################
getPayloadData.py \
    --plugin pluginSiStripBadStrip_PayloadInspector \
    --plot plot_SiStripBadStripFractionTH2PolyTkMap \
    --tag SiStripBadComponents_startupMC_for2017_v1_mc \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test ;

getPayloadData.py \
    --plugin pluginSiStripLorentzAngle_PayloadInspector \
    --plot plot_SiStripLorentzAngleTH2PolyTkMap \
    --tag  SiStripLorentzAngleDeco_GR10_v1_prompt \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test ;
