import FWCore.ParameterSet.Config as cms

from ..tasks.caloTowersRecTask_cfi import *

caloTowersRec = cms.Sequence(caloTowersRecTask)
