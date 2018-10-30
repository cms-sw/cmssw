#!/bin/bash

DIRBASE="$CMSSW_BASE/src/Alignment/APEEstimation"


mkdir $CMSSW_BASE/src/Alignment/TrackerAlignment/hists/


mkdir $DIRBASE/hists/
mkdir $DIRBASE/hists/workingArea/
mkdir $DIRBASE/hists/workingArea/apeObjects/
mkdir $DIRBASE/test/batch/workingArea/
mkdir $DIRBASE/test/autoSubmitter/workingArea/
mkdir $DIRBASE/test/cfgTemplateDesign/workingArea/
mkdir $DIRBASE/test/cfgTemplateMc/workingArea/
mkdir $DIRBASE/test/cfgTemplateData/workingArea/
#mkdir $DIRBASE/test/cfgTemplateParticleGun/workingArea/



cp $DIRBASE/test/cfgTemplate/createStep2.bash $DIRBASE/test/cfgTemplateDesign/createStep2.bash
cp $DIRBASE/test/cfgTemplate/startStep1.bash $DIRBASE/test/cfgTemplateDesign/startStep1.bash
cp $DIRBASE/test/cfgTemplate/startStep2.bash $DIRBASE/test/cfgTemplateDesign/startStep2.bash


cp $DIRBASE/test/cfgTemplate/createStep2.bash $DIRBASE/test/cfgTemplateMc/createStep2.bash
cp $DIRBASE/test/cfgTemplate/startStep1.bash $DIRBASE/test/cfgTemplateMc/startStep1.bash
cp $DIRBASE/test/cfgTemplate/startStep2.bash $DIRBASE/test/cfgTemplateMc/startStep2.bash


cp $DIRBASE/test/cfgTemplate/createStep2.bash $DIRBASE/test/cfgTemplateData/createStep2.bash
cp $DIRBASE/test/cfgTemplate/startStep1.bash $DIRBASE/test/cfgTemplateData/startStep1.bash
cp $DIRBASE/test/cfgTemplate/startStep2.bash $DIRBASE/test/cfgTemplateData/startStep2.bash


#cp $DIRBASE/test/cfgTemplate/createStep2.bash $DIRBASE/test/cfgTemplateParticleGun/createStep2.bash
#cp $DIRBASE/test/cfgTemplate/startStep1.bash $DIRBASE/test/cfgTemplateParticleGun/startStep1.bash
#cp $DIRBASE/test/cfgTemplate/startStep2.bash $DIRBASE/test/cfgTemplateParticleGun/startStep2.bash



## INFO: To run TrackListGenerator on AOD, need to comment in
## /Alignment/CommonAlignmentProducer/plugins/AlignmentTrackSelectorModule.cc
## the following line, since TrackExtra collection is not stored there:
## #include "CommonTools/RecoAlgos/interface/TrackSelector.h"



