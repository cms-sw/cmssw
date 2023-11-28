import FWCore.ParameterSet.Config as cms

from ..modules.ak4CaloJetsForTrk_cfi import *
from ..modules.hltPhase2TowerMakerForAll_cfi import *
from ..modules.firstStepPrimaryVerticesUnsorted_cfi import *

initialStepPVTask = cms.Task(
    ak4CaloJetsForTrk,
    hltPhase2TowerMakerForAll,
    firstStepPrimaryVerticesUnsorted,
)
