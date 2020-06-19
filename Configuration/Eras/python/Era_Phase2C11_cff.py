import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
from Configuration.Eras.Modifier_phase2_hgcalV12_cff import phase2_hgcalV12

Phase2C11 = cms.ModifierChain(Phase2C9, phase2_hgcalV12)

