import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70

Run2_2017_trackingPhase1PU70 = cms.ModifierChain(Run2_2016, phase1Pixel, trackingPhase1PU70)

