import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0

Phase2C11M9 = cms.ModifierChain(Phase2C11, phase2_GE0)
