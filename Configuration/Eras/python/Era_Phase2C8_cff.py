import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C4_cff import Phase2C4
from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
from Configuration.Eras.Modifier_phase2_trackerV14_cff import phase2_trackerV14

Phase2C8 = cms.ModifierChain(Phase2C4, phase2_hgcalV10, phase2_trackerV14)

