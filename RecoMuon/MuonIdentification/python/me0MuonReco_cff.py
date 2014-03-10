import FWCore.ParameterSet.Config as cms

from FastSimulation.Muons.me0SegmentProducer_cfi import *
from RecoMuon.MuonIdentification.me0SegmentMatcher_cfi import *
from RecoMuon.MuonIdentification.me0MuonConverter_cfi import *

me0MuonReco = cms.Sequence(me0SegmentProducer*me0SegmentMatcher*me0MuonConverter)
