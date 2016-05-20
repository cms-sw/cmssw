import FWCore.ParameterSet.Config as cms

from RecoLocalMuon.GEMRecHit.gemRecHits_cfi import *
from RecoLocalMuon.GEMSegment.gemSegments_cfi import *

gemLocalReco = cms.Sequence(gemRecHits*gemSegments)
