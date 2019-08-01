import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose

Phase2C10 = cms.ModifierChain(Phase2C9, phase2_hfnose)

