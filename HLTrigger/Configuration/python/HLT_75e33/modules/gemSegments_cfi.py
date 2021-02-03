import FWCore.ParameterSet.Config as cms

gemSegments = cms.EDProducer("GEMSegmentProducer",
    algo_name = cms.string('GEMSegmentAlgorithm'),
    algo_pset = cms.PSet(
        GEMDebug = cms.untracked.bool(True),
        clusterOnlySameBXRecHits = cms.bool(True),
        dEtaChainBoxMax = cms.double(0.05),
        dPhiChainBoxMax = cms.double(0.02),
        dXclusBoxMax = cms.double(1.0),
        dYclusBoxMax = cms.double(5.0),
        maxRecHitsInCluster = cms.int32(4),
        minHitsPerSegment = cms.uint32(2),
        preClustering = cms.bool(True),
        preClusteringUseChaining = cms.bool(True)
    ),
    gemRecHitLabel = cms.InputTag("gemRecHits")
)
