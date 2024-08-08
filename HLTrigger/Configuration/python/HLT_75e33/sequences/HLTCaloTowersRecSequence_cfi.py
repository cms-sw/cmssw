import FWCore.ParameterSet.Config as cms

from ..modules.hltTowerMaker_cfi import *

HLTCaloTowersRecSequence = cms.Sequence(hltTowerMaker)
