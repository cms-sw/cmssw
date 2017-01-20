import FWCore.ParameterSet.Config as cms

from Configuration.Eras.ModifierChain_run2_2017_core_cff import run2_2017_core
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70

Run2_2017_trackingPhase1PU70 = cms.ModifierChain(run2_2017_core, trackingPhase1PU70)

