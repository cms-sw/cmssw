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

    getPayloadData.py  \
 	--plugin pluginTrackerSurfaceDeformations_PayloadInspector \
 	--plot plot_SurfaceDeformationParameter${i}TkMapDelta \
 	--tag TrackerSurafceDeformations_v1_express \
 	--time_type Run \
	--iovs '{"start_iov": "283024", "end_iov": "299685"}' \
  	--db Prod \
 	--test;
    
    mv *.png $W_DIR/results_surfaces/SurfaceDeformationParameter${i}TkMapDelta.png

    #*************************************************************************#

    getPayloadData.py  \
 	--plugin pluginTrackerSurfaceDeformations_PayloadInspector \
 	--plot plot_TrackerSurfaceDeformationsPar${i}Comparator \
 	--tag TrackerSurafceDeformations_v1_express \
 	--time_type Run \
	--iovs '{"start_iov": "283024", "end_iov": "299685"}' \
  	--db Prod \
 	--test;
    
    mv *.png $W_DIR/results_surfaces/TrackerSurfaceDeformationsPar${i}Comparator.png

done
