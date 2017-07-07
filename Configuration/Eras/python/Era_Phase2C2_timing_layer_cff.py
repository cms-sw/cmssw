import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C2_timing_cff import Phase2C2_timing
from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer

Phase2C2_timing_layer = cms.ModifierChain(Phase2C2_timing, phase2_timing_layer)

