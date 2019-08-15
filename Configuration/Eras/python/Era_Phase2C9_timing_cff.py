import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing

Phase2C9_timing = cms.ModifierChain(Phase2C9, phase2_timing)

