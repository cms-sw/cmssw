#!/bin/bash

# Save current working dir so img can be outputted there later
W_DIR=$(pwd);

# Set SCRAM architecture var
SCRAM_ARCH=slc6_amd64_gcc530; 
export SCRAM_ARCH;

cd $W_DIR;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;

mkdir -p $W_DIR/results_APE

matrixelements=(XX YY ZZ XY XZ YZ)

for i in "${matrixelements[@]}"
do
    echo "Processing: $i element"
    
    #*************************************************************************#

    getPayloadData.py  \
	--plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
	--plot plot_TrackerAlignmentErrorExtended${i}Summary \
	--tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs \
	--time_type Run \
	--iovs '{"start_iov": "298759", "end_iov": "298759"}' \
	--db Prod \
	--test;

    mv *.png $W_DIR/results_APE/TrackerAlignmentErrorExtended${i}Summary.png
    
    #*************************************************************************#

    getPayloadData.py  \
	--plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
	--plot plot_TrackerAlignmentErrorExtended${i}Value \
	--tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs \
	--time_type Run \
	--iovs '{"start_iov": "1", "end_iov": "1"}' \
	--db Prod \
	--test;

    #*************************************************************************#
    
    getPayloadData.py  \
	--plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
	--plot plot_TrackerAlignmentErrorExtended${i}TrackerMap \
	--tag  TrackerAlignmentExtendedErr_2009_v2_express_IOVs \
	--time_type Run \
	--iovs '{"start_iov": "298759", "end_iov": "298759"}' \
	--db Prod \
	--test;
    
    mv *.png $W_DIR/results_APE/TrackerAlignmentErrorExtended${i}TrackerMap.png
    
    #*************************************************************************#

    getPayloadData.py  \
	--plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
	--plot plot_TrackerAlignmentErrorExtended${i}Comparator \
	--tag TrackerAlignmentExtendedErrors_v9_offline_IOVs \
	--time_type Run \
	--iovs '{"start_iov": "283681", "end_iov": "303886"}' \
	--db Prod \
	--test;

    mv *.png $W_DIR/results_APE/TrackerAlignmentErrorExtended_${i}_Comparison.png

    #*************************************************************************#

done

partitions=(BPix FPix TIB TOB TID)

for i in "${partitions[@]}"
do 
    echo "Processing $i partition"

    getPayloadData.py  \
	--plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
	--plot plot_TrackerAlignmentErrorExtended${i}Detail \
	--tag TrackerAlignmentExtendedErr_2009_v2_express_IOVs \
	--time_type Run \
	--iovs '{"start_iov": "1", "end_iov": "1"}' \
	--db Prod \
	--test;

    mv *.png $W_DIR/results_APE/TrackerAlignmentErrorExtended${i}Detail.png

done

#*************************************************************************#
# test two tags comparison
#*************************************************************************#

getPayloadData.py \
    --plugin pluginTrackerAlignmentErrorExtended_PayloadInspector \
    --plot plot_TrackerAlignmentErrorExtendedXXComparatorTwoTags \
    --tag TrackerAlignmentExtendedErrors_2016_ultralegacymc_v1 \
    --tagtwo TrackerAlignmentExtendedErrors_2016_ultralegacymc_preVFP_v1 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test;
