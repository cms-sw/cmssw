import FWCore.ParameterSet.Config as cms

from ..modules.ancientMuonSeed_cfi import *

standAloneMuonSeedsTask = cms.Task(ancientMuonSeed)
