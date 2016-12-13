import FWCore.ParameterSet.Config as cms

me0Segments = cms.EDProducer("ME0SegmentProducer",
    me0RecHitLabel = cms.InputTag("me0RecHits"),
    algo_name = cms.string("ME0SegmentAlgorithm"),                             
    algo_pset = cms.PSet(
        ME0Debug = cms.untracked.bool(True),
        minHitsPerSegment = cms.uint32(3),
        preClustering = cms.bool(True),
        dXclusBoxMax = cms.double(1.),
        dYclusBoxMax = cms.double(5.),
        preClusteringUseChaining = cms.bool(True),
        dPhiChainBoxMax = cms.double(.02),
        dEtaChainBoxMax = cms.double(.15),
        dTimeChainBoxMax = cms.double(15.0), # 1ns, +/- time to fly through 30cm thick ME0
        maxRecHitsInCluster = cms.int32(6)
    )
)
