import FWCore.ParameterSet.Config as cms

from RecoLocalMuon.GEMSegment.ME0SegmentsAlgorithm_cfi import *
from RecoLocalMuon.GEMSegment.ME0SegmentsRU_cfi import *

me0Segments = cms.EDProducer("ME0SegmentProducer",
    # Define input
    me0RecHitLabel = cms.InputTag("me0RecHits"),
    # Choice of the building algo: 1 Average, 2 RU, ...
    algo_type = cms.int32(2),
    # std::vector<edm::ParameterSet>
    algo_psets = cms.VPSet(
        cms.PSet(
            ME0SegmentAlgorithm
        ), 
        cms.PSet(
            ME0SegAlgoRU
        )

     )
)

