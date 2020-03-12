import FWCore.ParameterSet.Config as cms

#### PF CLUSTER PRESHOWER ####

#cleaning

#seeding
_localMaxSeeds_PS = cms.PSet(
    algoName = cms.string("LocalMaximumSeedFinder"),
    ### seed finding parameters    
    thresholdsByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("PS1"),
              seedingThreshold = cms.double(1.2e-4),
              seedingThresholdPt = cms.double(0.0)
              ),
    cms.PSet( detector = cms.string("PS2"),
              seedingThreshold = cms.double(1.2e-4),
              seedingThresholdPt = cms.double(0.0)
              )
    ),
    nNeighbours = cms.int32(4)
)

#topo clusters
_topoClusterizer_PS = cms.PSet(
    algoName = cms.string("Basic2DGenericTopoClusterizer"),
    thresholdsByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("PS1"),
              gatheringThreshold = cms.double(6e-5),
              gatheringThresholdPt = cms.double(0.0)
              ),
    cms.PSet( detector = cms.string("PS2"),
              gatheringThreshold = cms.double(6e-5),
              gatheringThresholdPt = cms.double(0.0)
              )
    ),    
    useCornerCells = cms.bool(False)
)

#position calc
_positionCalcPS_all_nodepth = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
    ##
    minFractionInCalc = cms.double(1e-9),
    posCalcNCrystals = cms.int32(-1),
    logWeightDenominator = cms.double(6e-5), # same as gathering threshold
    minAllowedNormalization = cms.double(1e-9)
)

#pf clustering
_pfClusterizer_PS = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowClusterizer"),
    #pf clustering parameters
    minFractionToKeep = cms.double(1e-7),
    positionCalc = _positionCalcPS_all_nodepth,
    showerSigma = cms.double(0.3),
    stoppingTolerance = cms.double(1e-8),
    maxIterations = cms.uint32(50),
    excludeOtherSeeds = cms.bool(True),
    minFracTot = cms.double(1e-20), ## numerical stabilization
    recHitEnergyNorms = cms.VPSet(
    cms.PSet( detector = cms.string("PS1"),
              recHitEnergyNorm = cms.double(6e-5)
              ),
    cms.PSet( detector = cms.string("PS2"),
              recHitEnergyNorm = cms.double(6e-5)
              )
    )
)

particleFlowClusterPS = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitPS"),
    recHitCleaners = cms.VPSet(),
    seedCleaners = cms.VPSet(),
    seedFinder = _localMaxSeeds_PS,
    initialClusteringStep = _topoClusterizer_PS,
    pfClusterBuilder = _pfClusterizer_PS,
    positionReCalc = cms.PSet(),
    energyCorrector = cms.PSet()
    )

