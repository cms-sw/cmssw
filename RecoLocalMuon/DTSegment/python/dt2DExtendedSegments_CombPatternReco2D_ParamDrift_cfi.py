import FWCore.ParameterSet.Config as cms

# module to build 2d extended segments
from RecoLocalMuon.DTSegment.DTCombinatorialPatternReco2DAlgo_ParamDrift_cfi import *
dt2DExtendedSegments = cms.EDProducer("DTRecSegment2DExtendedProducer",
    DTCombinatorialPatternReco2DAlgo_ParamDrift,
    debug = cms.untracked.bool(False),
    recClusLabel = cms.InputTag("dt1DClusters"),
    recHits1DLabel = cms.InputTag("dt1DRecHits")
)
# foo bar baz
