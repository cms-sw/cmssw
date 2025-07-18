import FWCore.ParameterSet.Config as cms

from ..modules.hltAk4CaloJetsForTrk_cfi import *
from ..modules.hltFirstStepPrimaryVerticesUnsorted_cfi import *
from ..modules.hltPhase2TowerMakerForAll_cfi import *

HLTInitialStepPVSequence = cms.Sequence(hltFirstStepPrimaryVerticesUnsorted+hltPhase2TowerMakerForAll+hltAk4CaloJetsForTrk)
