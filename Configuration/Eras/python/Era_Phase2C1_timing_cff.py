import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C1_cff import Phase2C1
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing

Phase2C1_timing = cms.ModifierChain(Phase2C1, phase2_timing)

