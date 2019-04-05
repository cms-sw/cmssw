import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C8_cff import Phase2C8
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing

Phase2C8_timing = cms.ModifierChain(Phase2C8, phase2_timing)

