import FWCore.ParameterSet.Config as cms

from ..modules.ak4CaloJetsForTrk_cfi import *
from ..modules.firstStepPrimaryVerticesUnsorted_cfi import *
from ..modules.hltPhase2TowerMakerForAll_cfi import *

initialStepPVSequence = cms.Sequence(firstStepPrimaryVerticesUnsorted+hltPhase2TowerMakerForAll+ak4CaloJetsForTrk)
