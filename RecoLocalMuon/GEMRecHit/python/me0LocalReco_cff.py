import FWCore.ParameterSet.Config as cms

from RecoLocalMuon.GEMRecHit.me0RecHits_cfi import *
from RecoLocalMuon.GEMSegment.me0Segments_cfi import *

me0LocalRecoTask = cms.Task(me0RecHits,me0Segments)
me0LocalReco = cms.Sequence(me0LocalRecoTask)
# foo bar baz
# M9Pl7BRYVaHg7
# QeXnE6zv9lJm3
