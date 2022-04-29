import FWCore.ParameterSet.Config as cms

particleFlowClusterHO = cms.EDProducer("PFClusterProducer",
    energyCorrector = cms.PSet(

    ),
    initialClusteringStep = cms.PSet(
        algoName = cms.string('Basic2DGenericTopoClusterizer'),
        thresholdsByDetector = cms.VPSet(
            cms.PSet(
                detector = cms.string('HCAL_BARREL2_RING0'),
                gatheringThreshold = cms.double(0.05),
                gatheringThresholdPt = cms.double(0.0)
            ),
            cms.PSet(
                detector = cms.string('HCAL_BARREL2_RING1'),
                gatheringThreshold = cms.double(0.05),
                gatheringThresholdPt = cms.double(0.0)
            )
        ),
        useCornerCells = cms.bool(True)
    ),
    pfClusterBuilder = cms.PSet(
        algoName = cms.string('Basic2DGenericPFlowClusterizer'),
        allCellsPositionCalc = cms.PSet(
            algoName = cms.string('Basic2DGenericPFlowPositionCalc'),
            logWeightDenominator = cms.double(0.05),
            minAllowedNormalization = cms.double(1e-09),
            minFractionInCalc = cms.double(1e-09),
            posCalcNCrystals = cms.int32(-1)
        ),
        excludeOtherSeeds = cms.bool(True),
        maxIterations = cms.uint32(50),
        minFracTot = cms.double(1e-20),
        minFractionToKeep = cms.double(1e-07),
        positionCalc = cms.PSet(
            algoName = cms.string('Basic2DGenericPFlowPositionCalc'),
            logWeightDenominator = cms.double(0.05),
            minAllowedNormalization = cms.double(1e-09),
            minFractionInCalc = cms.double(1e-09),
            posCalcNCrystals = cms.int32(5)
        ),
        recHitEnergyNorms = cms.VPSet(
            cms.PSet(
                detector = cms.string('HCAL_BARREL2_RING0'),
                recHitEnergyNorm = cms.double(0.05)
            ),
            cms.PSet(
                detector = cms.string('HCAL_BARREL2_RING1'),
                recHitEnergyNorm = cms.double(0.05)
            )
        ),
        showerSigma = cms.double(10.0),
        stoppingTolerance = cms.double(1e-08)
    ),
    positionReCalc = cms.PSet(

    ),
    recHitCleaners = cms.VPSet(),
    recHitsSource = cms.InputTag("particleFlowRecHitHO"),
    seedCleaners = cms.VPSet(),
    seedFinder = cms.PSet(
        algoName = cms.string('LocalMaximumSeedFinder'),
        nNeighbours = cms.int32(4),
        thresholdsByDetector = cms.VPSet(
            cms.PSet(
                detector = cms.string('HCAL_BARREL2_RING0'),
                seedingThreshold = cms.double(0.08),
                seedingThresholdPt = cms.double(0.0)
            ),
            cms.PSet(
                detector = cms.string('HCAL_BARREL2_RING1'),
                seedingThreshold = cms.double(0.08),
                seedingThresholdPt = cms.double(0.0)
            )
        )
    )
)
