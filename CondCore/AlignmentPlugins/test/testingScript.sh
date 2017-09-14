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

# /afs/cern.ch/user/c/condbpro/public/BROWSER_PI//getPayloadData.py  \
#     --plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
#     --plot plot_TrackerAlignmentErrorExtendedXValue \
#     --tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs\
#     --time_type Run \
#     --iovs '{"start_iov": "298450", "end_iov": "298450"}' \
#     --db Prod \
#     --test;

# /afs/cern.ch/user/c/condbpro/public/BROWSER_PI//getPayloadData.py  \
#     --plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
#     --plot plot_TrackerAlignmentErrorExtendedYValue \
#     --tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs\
#     --time_type Run \
#     --iovs '{"start_iov": "298450", "end_iov": "298450"}' \
#     --db Prod \
#     --test;

# /afs/cern.ch/user/c/condbpro/public/BROWSER_PI//getPayloadData.py  \
#     --plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
#     --plot plot_TrackerAlignmentErrorExtendedZValue \
#     --tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs\
#     --time_type Run \
#     --iovs '{"start_iov": "298450", "end_iov": "298450"}' \
#     --db Prod \
#     --test;

#*************************************************************************#

/afs/cern.ch/user/c/condbpro/public/BROWSER_PI//getPayloadData.py  \
    --plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
    --plot plot_TrackerAlignmentErrorExtendedXXSummary \
    --tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs\
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;

mv *.png $W_DIR/results/TrackerAlignmentErrorExtendedXXSummary.png

# #*************************************************************************#

# /afs/cern.ch/user/c/condbpro/public/BROWSER_PI//getPayloadData.py  \
#     --plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
#     --plot plot_TrackerAlignmentErrorExtendedXXTrackerMap \
#     --tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs\
#     --time_type Run \
#     --iovs '{"start_iov": "1", "end_iov": "1"}' \
#     --db Prod \
#     --test;

# mv *.png $W_DIR/results/TrackerAlignmentErrorExtendedXXTrackerMap.png

# #*************************************************************************#

# /afs/cern.ch/user/c/condbpro/public/BROWSER_PI//getPayloadData.py  \
#     --plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
#     --plot plot_TrackerAlignmentErrorExtendedYYTrackerMap \
#     --tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs\
#     --time_type Run \
#     --iovs '{"start_iov": "1", "end_iov": "1"}' \
#     --db Prod \
#     --test;

# mv *.png $W_DIR/results/TrackerAlignmentErrorExtendedYYTrackerMap.png

# #*************************************************************************#

# /afs/cern.ch/user/c/condbpro/public/BROWSER_PI//getPayloadData.py  \
#     --plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
#     --plot plot_TrackerAlignmentErrorExtendedZZTrackerMap \
#     --tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs\
#     --time_type Run \
#     --iovs '{"start_iov": "1", "end_iov": "1"}' \
#     --db Prod \
#     --test;

# mv *.png $W_DIR/results/TrackerAlignmentErrorExtendedZZTrackerMap.png

# #*************************************************************************#

# /afs/cern.ch/user/c/condbpro/public/BROWSER_PI//getPayloadData.py  \
#     --plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
#     --plot plot_TrackerAlignmentErrorExtendedXYTrackerMap \
#     --tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs\
#     --time_type Run \
#     --iovs '{"start_iov": "1", "end_iov": "1"}' \
#     --db Prod \
#     --test;

# mv *.png $W_DIR/results/TrackerAlignmentErrorExtendedXYTrackerMap.png

# #*************************************************************************#

# /afs/cern.ch/user/c/condbpro/public/BROWSER_PI//getPayloadData.py  \
#     --plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
#     --plot plot_TrackerAlignmentErrorExtendedXZTrackerMap \
#     --tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs\
#     --time_type Run \
#     --iovs '{"start_iov": "1", "end_iov": "1"}' \
#     --db Prod \
#     --test;

# mv *.png $W_DIR/results/TrackerAlignmentErrorExtendedXZTrackerMap.png

# #*************************************************************************#

# /afs/cern.ch/user/c/condbpro/public/BROWSER_PI//getPayloadData.py  \
#     --plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
#     --plot plot_TrackerAlignmentErrorExtendedYZTrackerMap \
#     --tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs\
#     --time_type Run \
#     --iovs '{"start_iov": "1", "end_iov": "1"}' \
#     --db Prod \
#     --test;

# mv *.png $W_DIR/results/TrackerAlignmentErrorExtendedYZTrackerMap.png

# #*************************************************************************#