import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2_cff import Phase2
from Configuration.Eras.Modifier_phase2_tracker_postTDR_cff import phase2_tracker_postTDR

Phase2_tracker_postTDR = cms.ModifierChain(Phase2, phase2_tracker_postTDR)

