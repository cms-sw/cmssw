import FWCore.ParameterSet.Config as cms

particleFlowClusterPS = cms.EDProducer("PFClusterProducer",
    energyCorrector = cms.PSet(

    ),
    initialClusteringStep = cms.PSet(
        algoName = cms.string('Basic2DGenericTopoClusterizer'),
        thresholdsByDetector = cms.VPSet(
            cms.PSet(
                detector = cms.string('PS1'),
                gatheringThreshold = cms.double(6e-05),
                gatheringThresholdPt = cms.double(0.0)
            ),
            cms.PSet(
                detector = cms.string('PS2'),
                gatheringThreshold = cms.double(6e-05),
                gatheringThresholdPt = cms.double(0.0)
            )
        ),
        useCornerCells = cms.bool(False)
    ),
    pfClusterBuilder = cms.PSet(
        algoName = cms.string('Basic2DGenericPFlowClusterizer'),
        excludeOtherSeeds = cms.bool(True),
        maxIterations = cms.uint32(50),
        minFracTot = cms.double(1e-20),
        minFractionToKeep = cms.double(1e-07),
        positionCalc = cms.PSet(
            algoName = cms.string('Basic2DGenericPFlowPositionCalc'),
            logWeightDenominator = cms.double(6e-05),
            minAllowedNormalization = cms.double(1e-09),
            minFractionInCalc = cms.double(1e-09),
            posCalcNCrystals = cms.int32(-1)
        ),
        recHitEnergyNorms = cms.VPSet(
            cms.PSet(
                detector = cms.string('PS1'),
                recHitEnergyNorm = cms.double(6e-05)
            ),
            cms.PSet(
                detector = cms.string('PS2'),
                recHitEnergyNorm = cms.double(6e-05)
            )
        ),
        showerSigma = cms.double(0.3),
        stoppingTolerance = cms.double(1e-08)
    ),
    positionReCalc = cms.PSet(

    ),
    recHitCleaners = cms.VPSet(),
    recHitsSource = cms.InputTag("particleFlowRecHitPS"),
    seedCleaners = cms.VPSet(),
    seedFinder = cms.PSet(
        algoName = cms.string('LocalMaximumSeedFinder'),
        nNeighbours = cms.int32(4),
        thresholdsByDetector = cms.VPSet(
            cms.PSet(
                detector = cms.string('PS1'),
                seedingThreshold = cms.double(0.00012),
                seedingThresholdPt = cms.double(0.0)
            ),
            cms.PSet(
                detector = cms.string('PS2'),
                seedingThreshold = cms.double(0.00012),
                seedingThresholdPt = cms.double(0.0)
            )
        )
    )
)
