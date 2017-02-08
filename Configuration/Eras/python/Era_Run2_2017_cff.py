import FWCore.ParameterSet.Config as cms

from Configuration.Eras.ModifierChain_run2_2017_core_cff import run2_2017_core
from Configuration.Eras.Modifier_trackingPhase1QuadProp_cff import trackingPhase1QuadProp
from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017

Run2_2017 = cms.ModifierChain(run2_2017_core, trackingPhase1QuadProp, run2_GEM_2017)

