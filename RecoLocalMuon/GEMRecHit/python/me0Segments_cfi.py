import FWCore.ParameterSet.Config as cms

me0Segments = cms.EDProducer("CSCSegmentProducer",
 #define input
 inputObjects = cms.InputTag("me0RecHits"),
 algo_name = cms.string("ME0SegAlgoMM"),                             
 algo_pset = cms.PSet(
     ME0Debug = cms.bool(true),
     minHitsPerSegment = cms.uint32(3),
     preClustering = cms.bool(true),
     dXclusBoxMax = cms.double(1.),
     dYclusBoxMax = cms.double(5.),
     preClusteringUseChaining = cms.bool(true),
     dPhiChainBoxMax = cms.double(1.),
     dEtaChainBoxMax = cms.double(1.),
     maxRecHitsInCluster = cms.int(6)
     )
)
