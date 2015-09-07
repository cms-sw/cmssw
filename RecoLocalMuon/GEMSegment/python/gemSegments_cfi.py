import FWCore.ParameterSet.Config as cms

gemSegments = cms.EDProducer("GEMSegmentProducer",
    gemRecHitLabel = cms.InputTag("gemRecHits"),
    algo_name = cms.string("GEMSegAlgoPV"),                             
    algo_pset = cms.PSet(
        GEMDebug = cms.untracked.bool(True),
        minHitsPerSegment = cms.uint32(2),
        preClustering = cms.bool(True),
        dXclusBoxMax = cms.double(1.),
        dYclusBoxMax = cms.double(5.),
        preClusteringUseChaining = cms.bool(True),
        dPhiChainBoxMax = cms.double(.02),
        dEtaChainBoxMax = cms.double(.05),
        maxRecHitsInCluster = cms.int32(3)
    )
)
