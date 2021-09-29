import FWCore.ParameterSet.Config as cms

hltMe0Segments = cms.EDProducer("ME0SegmentProducer",
    algo_psets = cms.VPSet(
        cms.PSet(
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
        ),
        cms.PSet(
            algo_name = cms.string('ME0SegAlgoRU'),
            algo_pset = cms.PSet(
                allowWideSegments = cms.bool(True),
                doCollisions = cms.bool(True),
                maxChi2Additional = cms.double(100.0),
                maxChi2GoodSeg = cms.double(50),
                maxChi2Prune = cms.double(50),
                maxETASeeds = cms.double(0.1),
                maxPhiAdditional = cms.double(0.001096605744),
                maxPhiSeeds = cms.double(0.001096605744),
                maxTOFDiff = cms.double(25),
                minNumberOfHits = cms.uint32(4),
                requireCentralBX = cms.bool(True)
            )
        )
    ),
    algo_type = cms.int32(2),
    me0RecHitLabel = cms.InputTag("hltMe0RecHits")
)
