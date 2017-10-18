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

####################
# Test Gains
####################
/afs/cern.ch/user/c/condbpro/public/BROWSER_PI/getPayloadData.py \
    --plugin pluginSiStripApvGain_PayloadInspector \
    --plot plot_SiStripApvGainsByPartition \
    --tag SiStripApvGain_FromParticles_GR10_v1_express \
    --time_type Run \
    --iovs '{"start_iov": "286042", "end_iov": "286042"}' \
    --db Prod \
    --test;

######################
# Test Lorentz Angle
######################
/afs/cern.ch/user/c/condbpro/public/BROWSER_PI/getPayloadData.py \
    --plugin pluginSiStripLorentzAngle_PayloadInspector \
    --plot plot_SiStripLorentzAngleByPartition \
    --tag  SiStripLorentzAngleDeco_GR10_v1_prompt \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

######################
# Test Backplane correction
######################
/afs/cern.ch/user/c/condbpro/public/BROWSER_PI/getPayloadData.py \
    --plugin pluginSiStripBackPlaneCorrection_PayloadInspector \
    --plot plot_SiStripBackPlaneCorrectionByPartition \
    --tag SiStripBackPlaneCorrection_deco_GR10_v1_express \
    --time_type Run \
    --iovs '{"start_iov": "153690", "end_iov": "153690"}' \
    --db Prod \
    --test;

