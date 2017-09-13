#!/bin/bash

# Save current working dir so img can be outputted there later
W_DIR=$(pwd);

# Set SCRAM architecture var
SCRAM_ARCH=slc6_amd64_gcc530; 
export SCRAM_ARCH;

cd $W_DIR;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;

$W_DIR/getPayloadData.py  \
    --plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
    --plot plot_TrackerAlignmentErrorExtendedXValue \
    --tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs\
    --time_type Run \
    --iovs '{"start_iov": "298450", "end_iov": "298450"}' \
    --db Prod \
    --test;

$W_DIR/getPayloadData.py  \
    --plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
    --plot plot_TrackerAlignmentErrorExtendedYValue \
    --tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs\
    --time_type Run \
    --iovs '{"start_iov": "298450", "end_iov": "298450"}' \
    --db Prod \
    --test;

$W_DIR/getPayloadData.py  \
    --plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
    --plot plot_TrackerAlignmentErrorExtendedZValue \
    --tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs\
    --time_type Run \
    --iovs '{"start_iov": "298450", "end_iov": "298450"}' \
    --db Prod \
    --test;