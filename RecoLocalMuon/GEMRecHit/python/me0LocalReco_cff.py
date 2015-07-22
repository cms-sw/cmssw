import FWCore.ParameterSet.Config as cms

from RecoLocalMuon.GEMRecHit.me0RecHits_cfi import *
from RecoLocalMuon.GEMRecHit.me0Segments_cfi import *

me0LocalReco = cms.Sequence(me0RecHits*me0Segments)
