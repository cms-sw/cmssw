import FWCore.ParameterSet.Config as cms

from RecoLocalMuon.GEMRecHit.gemRecHits_cfi import *
from RecoLocalMuon.GEMSegment.gemSegments_cfi import *

gemLocalRecoTask = cms.Task(gemRecHits,gemSegments)
gemLocalReco = cms.Sequence(gemLocalRecoTask)
