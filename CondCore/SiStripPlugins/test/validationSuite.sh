#!/bin/bash

# Save current working dir so img can be outputted there later
W_DIR=$(pwd);

# Set SCRAM architecture var
SCRAM_ARCH=slc6_amd64_gcc530; 
export SCRAM_ARCH;

cd $W_DIR;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;

mkdir -p $W_DIR/results

#-----------------------------------------------------------

$W_DIR/getPayloadData.py  \
    --plugin pluginSiStripApvGain_PayloadInspector \
    --plot plot_SiStripApvGainsByPartition \
    --tag SiStripApvGain_FromParticles_GR10_v1_express \
    --time_type Run \
    --iovs '{"start_iov": "300577", "end_iov": "300577"}' \
    --db Prod \
    --image_plot True \
    --test;

mv *.png $W_DIR/results/SiStripApvGainsByPartition.png

#-----------------------------------------------------------

$W_DIR/getPayloadData.py  \
    --plugin pluginSiStripApvGain_PayloadInspector \
    --plot plot_SiStripApvGainsComparatorByPartition \
    --tag SiStripApvGain_FromParticles_GR10_v1_express \
    --time_type Run \
    --iovs '{"start_iov": "286042", "end_iov": "300577"}' \
    --db Prod \
    --image_plot True \
    --test;

mv *.png $W_DIR/results/SiStripApvGainsComparatorByPartition.png

#-----------------------------------------------------------

$W_DIR/getPayloadData.py  \
    --plugin pluginSiStripApvGain_PayloadInspector \
    --plot plot_SiStripApvGainsTest \
    --tag SiStripApvGain_FromParticles_GR10_v1_express \
    --time_type Run \
    --iovs '{"start_iov": "300577", "end_iov": "300577"}' \
    --db Prod \
    --image_plot True \
    --test;

#-----------------------------------------------------------

$W_DIR/getPayloadData.py  \
    --plugin pluginSiStripApvGain_PayloadInspector \
    --plot plot_SiStripApvGainsDefaultTrackerMap \
    --tag  SiStripApvGain_FromParticles_GR10_v1_express \
    --time_type Run \
    --iovs '{"start_iov": "300577", "end_iov": "300577"}' \
    --db Prod \
    --image_plot True \
    --test;

mv *.png $W_DIR/results/SiStripApvGainsDefaultTrackerMap.png

#-----------------------------------------------------------

$W_DIR/getPayloadData.py  \
    --plugin pluginSiStripApvGain_PayloadInspector \
    --plot plot_SiStripApvGainsComparator \
    --tag SiStripApvGain_FromParticles_GR10_v1_express \
    --time_type Run \
    --iovs '{"start_iov": "286042", "end_iov": "300577"}' \
    --db Prod \
    --image_plot True \
    --test;

mv *.png $W_DIR/results/SiStripApvGainsComparator.png

#-----------------------------------------------------------