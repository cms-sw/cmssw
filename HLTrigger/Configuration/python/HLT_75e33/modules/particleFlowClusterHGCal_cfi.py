import FWCore.ParameterSet.Config as cms

particleFlowClusterHGCal = cms.EDProducer("PFClusterProducer",
    energyCorrector = cms.PSet(

    ),
    initialClusteringStep = cms.PSet(
        algoName = cms.string('PFClusterFromHGCalTrackster'),
        clusterSrc = cms.InputTag("hgcalMergeLayerClusters"),
        filterByTracksterIteration = cms.bool(True),
        filterByTracksterPID = cms.bool(False),
        filter_on_categories = cms.vint32(0, 1),
        filter_on_iterations = cms.vint32(0, 1),
        pid_threshold = cms.double(0.8),
        thresholdsByDetector = cms.VPSet(),
        tracksterSrc = cms.InputTag("ticlTrackstersMerge")
    ),
    pfClusterBuilder = cms.PSet(

    ),
    positionReCalc = cms.PSet(
        algoName = cms.string('Cluster3DPCACalculator'),
        minFractionInCalc = cms.double(1e-09),
        updateTiming = cms.bool(False)
    ),
    recHitCleaners = cms.VPSet(),
    recHitsSource = cms.InputTag("particleFlowRecHitHGC"),
    seedCleaners = cms.VPSet(),
    seedFinder = cms.PSet(
        algoName = cms.string('PassThruSeedFinder'),
        nNeighbours = cms.int32(8),
        thresholdsByDetector = cms.VPSet()
    )
)
