import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2_cff import Phase2
from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9

Phase2C4 = cms.ModifierChain(Phase2, phase2_hgcalV9)

