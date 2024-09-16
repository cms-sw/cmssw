import FWCore.ParameterSet.Config as cms

hltParticleFlowClusterHGCal = cms.EDProducer("PFClusterProducer",
    energyCorrector = cms.PSet(

    ),
    initialClusteringStep = cms.PSet(
        algoName = cms.string('PFClusterFromHGCalTrackster'),
        clusterSrc = cms.InputTag("hltHgcalMergeLayerClusters"),
        filterByTracksterIteration = cms.bool(True),
        filterByTracksterPID = cms.bool(False),
        filter_on_categories = cms.vint32(0, 1),
        filter_on_iterations = cms.vint32(0, 1),
        pid_threshold = cms.double(0.8),
        thresholdsByDetector = cms.VPSet(),
        tracksterSrc = cms.InputTag("hltTiclTrackstersMerge")
    ),
    pfClusterBuilder = cms.PSet(

    ),
    positionReCalc = cms.PSet(
        algoName = cms.string('Cluster3DPCACalculator'),
        minFractionInCalc = cms.double(1e-09),
        updateTiming = cms.bool(False)
    ),
    recHitCleaners = cms.VPSet(),
    recHitsSource = cms.InputTag("hltParticleFlowRecHitHGC"),
    seedCleaners = cms.VPSet(),
    seedFinder = cms.PSet(
        algoName = cms.string('PassThruSeedFinder'),
        nNeighbours = cms.int32(8),
        thresholdsByDetector = cms.VPSet()
    ),
    usePFThresholdsFromDB = cms.bool(False)
)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
ticl_v5.toModify(hltParticleFlowClusterHGCal.initialClusteringStep, tracksterSrc = "hltTiclCandidate")
