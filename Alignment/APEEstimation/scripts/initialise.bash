#!/bin/bash

DIRBASE="$CMSSW_BASE/src/Alignment/APEEstimation"

mkdir $DIRBASE/hists/
mkdir $DIRBASE/hists/workingArea/
mkdir $DIRBASE/hists/workingArea/apeObjects/
mkdir $DIRBASE/test/SkimProducer/workingArea/
mkdir $DIRBASE/test/autoSubmitter/workingArea/

## INFO: To run TrackListGenerator on AOD, need to comment in
## /Alignment/CommonAlignmentProducer/plugins/AlignmentTrackSelectorModule.cc
## the following line, since TrackExtra collection is not stored there:
## #include "CommonTools/RecoAlgos/interface/TrackSelector.h"



