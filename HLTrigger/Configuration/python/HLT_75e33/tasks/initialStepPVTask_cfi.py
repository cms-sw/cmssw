import FWCore.ParameterSet.Config as cms

from ..modules.ak4CaloJetsForTrk_cfi import *
from ..modules.hltTowerMakerForAll_cfi import *
from ..modules.firstStepPrimaryVerticesUnsorted_cfi import *

initialStepPVTask = cms.Task(
    ak4CaloJetsForTrk,
    hltTowerMakerForAll,
    firstStepPrimaryVerticesUnsorted,
)
