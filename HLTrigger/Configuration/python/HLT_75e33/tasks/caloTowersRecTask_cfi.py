import FWCore.ParameterSet.Config as cms

from ..modules.towerMaker_cfi import *
from ..modules.towerMakerWithHO_cfi import *

caloTowersRecTask = cms.Task(towerMaker, towerMakerWithHO)
