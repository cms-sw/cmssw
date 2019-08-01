import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C10_cff import Phase2C10
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing

Phase2C10_timing = cms.ModifierChain(Phase2C10, phase2_timing)

