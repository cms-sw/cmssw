import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C6_cff import Phase2C6
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing

Phase2C6_timing = cms.ModifierChain(Phase2C6, phase2_timing)

