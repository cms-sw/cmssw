import FWCore.ParameterSet.Config as cms

from RecoLocalMuon.GEMRecHit.me0RecHits_cfi import *
from RecoLocalMuon.GEMSegment.me0Segments_cfi import *

me0LocalRecoTask = cms.Task(me0RecHits,me0Segments)
me0LocalReco = cms.Sequence(me0LocalRecoTask)
