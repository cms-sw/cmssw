import FWCore.ParameterSet.Config as cms

ME0SegmentAlgorithm = cms.PSet(
    algo_name = cms.string('ME0SegmentAlgorithm'),
    algo_pset = cms.PSet(
        ME0Debug = cms.untracked.bool(True),
        dEtaChainBoxMax = cms.double(0.05),
        dPhiChainBoxMax = cms.double(0.02),
        dTimeChainBoxMax = cms.double(15.0),
        dXclusBoxMax = cms.double(1.0),
        dYclusBoxMax = cms.double(5.0),
        maxRecHitsInCluster = cms.int32(6),
        minHitsPerSegment = cms.uint32(3),
        preClustering = cms.bool(True),
        preClusteringUseChaining = cms.bool(True)
    )
)