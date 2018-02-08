#!/bin/bash

# Save current working dir so img can be outputted there later
W_DIR=$(pwd);

# Set SCRAM architecture var
SCRAM_ARCH=slc6_amd64_gcc530; 
export SCRAM_ARCH;

cd $W_DIR;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;

mkdir -p $W_DIR/results_surfaces

getPayloadData.py  \
 	--plugin pluginTrackerSurfaceDeformations_PayloadInspector \
 	--plot plot_TrackerSurfaceDeformationsTest \
 	--tag  TrackerSurafceDeformations_v1_express \
 	--time_type Run \
	--iovs '{"start_iov": "299685", "end_iov": "299685"}' \
  	--db Prod \
 	--test;

#*************************************************************************#
elements=(BPix FPix TIB TOB TID TEC)

for i in "${elements[@]}"
do
    echo "Processing: $i partition"
    
    getPayloadData.py  \
 	--plugin pluginTrackerSurfaceDeformations_PayloadInspector \
 	--plot plot_${i}SurfaceDeformationsSummary \
 	--tag TrackerSurafceDeformations_v1_express \
 	--time_type Run \
	--iovs '{"start_iov": "299685", "end_iov": "299685"}' \
  	--db Prod \
 	--test;
    
    mv *.png $W_DIR/results_surfaces/SurfaceDeformationSummary${i}.png
done

#*************************************************************************#
elements=(BPix FPix TIB TOB TID TEC)

for i in "${elements[@]}"
do
    echo "Processing: $i partition"
    
    getPayloadData.py  \
 	--plugin pluginTrackerSurfaceDeformations_PayloadInspector \
 	--plot plot_${i}SurfaceDeformationsComparison \
 	--tag TrackerSurafceDeformations_v1_express \
 	--time_type Run \
	--iovs '{"start_iov": "283024", "end_iov": "299685"}' \
  	--db Prod \
 	--test;
    
    mv *.png $W_DIR/results_surfaces/SurfaceDeformationComparison${i}.png
done

#*************************************************************************#
for i in {0..12}
do 
    echo "Processing: $i parameter"
    
    getPayloadData.py  \
 	--plugin pluginTrackerSurfaceDeformations_PayloadInspector \
 	--plot plot_SurfaceDeformationParameter${i}TrackerMap \
 	--tag TrackerSurafceDeformations_v1_express \
 	--time_type Run \
	--iovs '{"start_iov": "299685", "end_iov": "299685"}' \
  	--db Prod \
 	--test;
    
    mv *.png $W_DIR/results_surfaces/SurfaceDeformationTrackerMapParameter_${i}.png

done


#*************************************************************************#
for i in {0..12}
do 
    echo "Processing: $i parameter"
    
    getPayloadData.py  \
 	--plugin pluginTrackerSurfaceDeformations_PayloadInspector \
 	--plot plot_SurfaceDeformationParameter${i}TkMapDelta \
 	--tag TrackerSurafceDeformations_v1_express \
 	--time_type Run \
	--iovs '{"start_iov": "283024", "end_iov": "299685"}' \
  	--db Prod \
 	--test;
    
    mv *.png $W_DIR/results_surfaces/SurfaceDeformationParameter${i}TkMapDelta.png

done
