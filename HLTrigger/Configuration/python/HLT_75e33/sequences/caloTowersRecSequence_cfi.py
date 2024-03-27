import FWCore.ParameterSet.Config as cms

from ..modules.towerMaker_cfi import *

caloTowersRecSequence = cms.Sequence(towerMaker)
