import FWCore.ParameterSet.Config as cms

me0Segments = cms.EDProducer("ME0SegmentProducer",
    me0RecHitLabel = cms.InputTag("me0RecHits"),
    algo_name = cms.string("ME0SegAlgoMM"),                             
    algo_pset = cms.PSet(
        ME0Debug = cms.untracked.bool(True),
        minHitsPerSegment = cms.uint32(3),
        preClustering = cms.bool(True),
        dXclusBoxMax = cms.double(1.),
        dYclusBoxMax = cms.double(5.),
        preClusteringUseChaining = cms.bool(True),
        dPhiChainBoxMax = cms.double(1.),
        dEtaChainBoxMax = cms.double(1.),
        maxRecHitsInCluster = cms.int32(6)
    )
)
