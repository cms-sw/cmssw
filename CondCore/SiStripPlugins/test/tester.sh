#!/bin/bash

# Save current working dir so img can be outputted there later
W_DIR=$(pwd);

# Set SCRAM architecture var
SCRAM_ARCH=slc6_amd64_gcc530; 
export SCRAM_ARCH;

cd $W_DIR;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;

# $W_DIR/getPayloadData.py  \
#     --plugin pluginSiStripApvGain_PayloadInspector \
#     --plot plot_SiStripApvGainsByPartition \
#     --tag SiStripApvGain \
#     --time_type Run \
#     --iovs '{"start_iov": "299061", "end_iov": "299061"}' \
#     --db sqlite_file:toCheck.db \
#     --image_plot True \
#     --test;

# $W_DIR/getPayloadData.py  \
#     --plugin pluginSiStripApvGain_PayloadInspector \
#     --plot plot_SiStripApvGainsComparatorByPartition \
#     --tag SiStripApvGain \
#     --time_type Run \
#     --iovs '{"start_iov": "286042", "end_iov": "299649"}' \
#     --db sqlite_file:toCheck.db \
#     --image_plot True \
#     --test;

# $W_DIR/getPayloadData.py  \
#     --plugin pluginSiStripApvGain_PayloadInspector \
#     --plot plot_SiStripApvGainsTest \
#     --tag SiStripApvGain \
#     --time_type Run \
#     --iovs '{"start_iov": "286042", "end_iov": "286042"}' \
#     --db sqlite_file:toCheck.db \
#     --image_plot True \
#     --test;


# $W_DIR/getPayloadData.py  \
#     --plugin pluginSiStripApvGain_PayloadInspector \
#     --plot plot_SiStripApvGainsDefaultTrackerMap \
#     --tag SiStripApvGain \
#     --time_type Run \
#     --iovs '{"start_iov": "299061", "end_iov": "299061"}' \
#     --db sqlite_file:toCheck.db \
#     --image_plot True \
#     --test;

# $W_DIR/getPayloadData.py  \
#     --plugin pluginSiStripApvGain_PayloadInspector \
#     --plot plot_SiStripApvGainsComparator \
#     --tag SiStripApvGain \
#     --time_type Run \
#     --iovs '{"start_iov": "286042", "end_iov": "299649"}' \
#     --db sqlite_file:toCheck.db \
#     --image_plot True \
#     --test;

# $W_DIR/getPayloadData.py  \
#     --plugin pluginSiStripApvGain_PayloadInspector \
#     --plot plot_SiStripApvGainsDefaultTrackerMap \
#     --tag  SiStripApvGain_GR10_v1_hlt\
#     --time_type Run \
#     --iovs '{"start_iov": "298430", "end_iov": "298430"}' \
#     --db Prod \
#     --image_plot True \
#     --test;

$W_DIR/getPayloadData.py  \
    --plugin pluginSiStripApvGain_PayloadInspector \
    --plot plot_SiStripApvGainsDefaultTrackerMap \
    --tag  SiStripApvGain_FromParticles_GR10_v1_express \
    --time_type Run \
    --iovs '{"start_iov": "300577", "end_iov": "300577"}' \
    --db Prod \
    --image_plot True \
    --test;