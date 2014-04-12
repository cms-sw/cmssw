#!/usr/bin/bash

# Define the paths for running the jobs, storing all output files,
# and the location of the production system.

RUNDIR=/afs/hephy.at/scratch/e/ewidl/CMSSW_2_1_10/src/Alignment/KalmanAlignmentAlgorithm/test
MSSDIR=/afs/hephy.at/scratch/e/ewidl/CMSSW_2_1_10/src/Alignment/KalmanAlignmentAlgorithm/data
KAPSDIR=/afs/hephy.at/scratch/e/ewidl/CMSSW_2_1_10/src/Alignment/KalmanAlignmentAlgorithm/scripts

# Setup the production system.
export PATH=${PATH}:$KAPSDIR
source $KAPSDIR/kaps.sh

# Change to the working directory
cd $RUNDIR

#####################################################
#
# Run the alignment jobs
#
#####################################################

# Alignment of TOB and TEC dets
alignment_run $KAPSDIR/kaps_template.sh template.outer_tracker_dets_cfg.py data.muon_isolated.txt 10 OUTER_TRACKER_DETS $KAPSDIR/kaps_merge_template.sh $MSSDIR
fetch_output $MSSDIR outer-tracker-dets

# Alignment of TIB and TID layers
alignment_run $KAPSDIR/kaps_template.sh template.inner_tracker_layers_cfg.py data.cosmics.txt 10 INNER_TRACKER_LAYERS $KAPSDIR/kaps_merge_template.sh $MSSDIR
fetch_output $MSSDIR inner-tracker-layers

# Alignment of TIB and TID dets
alignment_run $KAPSDIR/kaps_template.sh template.inner_tracker_dets_cfg.py data.muon_isolated.txt 10 INNER_TRACKER_DETS $KAPSDIR/kaps_merge_template.sh $MSSDIR
fetch_output $MSSDIR inner-tracker-dets

# Alignment of TPB and TPE layers
alignment_run $KAPSDIR/kaps_template.sh template.pixel_tracker_layers_cfg.py data.muon_isolated.txt 10 PIXEL_TRACKER_LAYERS $KAPSDIR/kaps_merge_template.sh $MSSDIR
fetch_output $MSSDIR pixel-tracker-layers

# Alignment of TPB and TPE dets
alignment_run $KAPSDIR/kaps_template.sh template.pixel_tracker_dets_cfg.py data.muon_isolated.txt 10 PIXEL_TRACKER_DETS $KAPSDIR/kaps_merge_template.sh $MSSDIR
fetch_output $MSSDIR pixel-tracker-dets

# Alignment of TPB and TPE layers
alignment_run $KAPSDIR/kaps_template.sh template.pixel_tracker_layers_final_cfg.py data.zmumu.txt 1 PIXEL_TRACKER_LAYERS_FINAL $KAPSDIR/kaps_merge_template.sh $MSSDIR
fetch_output $MSSDIR pixel-tracker-layers-final
