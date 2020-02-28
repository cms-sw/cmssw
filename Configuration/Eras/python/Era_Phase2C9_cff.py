import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C8_cff import Phase2C8
from Configuration.Eras.Modifier_phase2_hgcalV11_cff import phase2_hgcalV11

Phase2C9 = cms.ModifierChain(Phase2C8, phase2_hgcalV11)

