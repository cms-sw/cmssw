import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C2_cff import Phase2C2
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing

Phase2C2_timing = cms.ModifierChain(Phase2C2, phase2_timing)

