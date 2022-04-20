import FWCore.ParameterSet.Config as cms

from ..modules.towerMaker_cfi import *

caloTowersRecTask = cms.Task(
    towerMaker
)
