import FWCore.ParameterSet.Config as cms

particleFlowClusterHGCal = cms.EDProducer("PFClusterProducer",
    energyCorrector = cms.PSet(

    ),
    initialClusteringStep = cms.PSet(
        algoName = cms.string('RealisticSimClusterMapper'),
        calibMaxEta = cms.double(3.0),
        calibMinEta = cms.double(1.4),
        egammaCalib = cms.vdouble(
            1.0, 1.0, 1.01, 1.01, 1.02,
            1.01, 1.01, 1.01
        ),
        exclusiveFraction = cms.double(0.6),
        hadronCalib = cms.vdouble(
            1.28, 1.28, 1.24, 1.19, 1.17,
            1.17, 1.17, 1.17
        ),
        invisibleFraction = cms.double(0.6),
        maxDforTimingSquared = cms.double(4.0),
        maxDistance = cms.double(10.0),
        maxDistanceFilter = cms.bool(True),
        minNHitsforTiming = cms.uint32(3),
        simClusterSrc = cms.InputTag("mix","MergedCaloTruth"),
        thresholdsByDetector = cms.VPSet(),
        timeOffset = cms.double(5),
        useMCFractionsForExclEnergy = cms.bool(False)
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
