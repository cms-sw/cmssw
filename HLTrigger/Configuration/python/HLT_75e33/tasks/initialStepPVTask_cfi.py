import FWCore.ParameterSet.Config as cms

from ..modules.ak4CaloJetsForTrk_cfi import *
from ..modules.caloTowerForTrk_cfi import *
from ..modules.firstStepPrimaryVerticesUnsorted_cfi import *

initialStepPVTask = cms.Task(
    ak4CaloJetsForTrk,
    caloTowerForTrk,
    firstStepPrimaryVerticesUnsorted,
)
