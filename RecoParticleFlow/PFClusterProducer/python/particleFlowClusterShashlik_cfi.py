import FWCore.ParameterSet.Config as cms

#### PF CLUSTER SHASHLIK ####

#cleaning
## no cleaning for Shashlik yet

#seeding
_localMaxSeeds_EK = cms.PSet(
    algoName = cms.string("LocalMaximumSeedFinder"),
    thresholdsByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("ECAL_ENDCAP"),
              seedingThreshold = cms.double(0.3),
              seedingThresholdPt = cms.double(0.075)
              )
    ),
    nNeighbours = cms.int32(8)
)

# topo clusterizer
_topoClusterizer_EK = cms.PSet(
    algoName = cms.string("Basic2DGenericTopoClusterizer"),
    thresholdsByDetector = cms.VPSet(   
    cms.PSet( detector = cms.string("ECAL_ENDCAP"),
              gatheringThreshold = cms.double(0.08),
              gatheringThresholdPt = cms.double(0.0)
              )
    ),
    useCornerCells = cms.bool(True)
)

#position calculations
_positionCalcECAL_all_nodepth = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
    ##
    minFractionInCalc = cms.double(1e-9),
    posCalcNCrystals = cms.int32(-1),
    logWeightDenominator = cms.double(0.08), # same as gathering threshold
    minAllowedNormalization = cms.double(1e-9),
    #timeResolutionCalcBarrel = _timeResolutionECALBarrel,
    #timeResolutionCalcEndcap = _timeResolutionECALEndcap,
)
_positionCalcECAL_3x3_nodepth = _positionCalcECAL_all_nodepth.clone(
    posCalcNCrystals = cms.int32(9)
)

# pf clustering
_pfClusterizer_EK = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowClusterizer"),
    #pf clustering parameters
    minFractionToKeep = cms.double(1e-7),
    positionCalc = _positionCalcECAL_3x3_nodepth,
    allCellsPositionCalc = _positionCalcECAL_all_nodepth,
    showerSigma = cms.double(1.42),
    stoppingTolerance = cms.double(1e-8),
    maxIterations = cms.uint32(50),
    excludeOtherSeeds = cms.bool(True),
    minFracTot = cms.double(1e-20), ## numerical stabilization
    recHitEnergyNorms = cms.VPSet(
    cms.PSet( detector = cms.string("ECAL_ENDCAP"),
              recHitEnergyNorm = cms.double(0.08)
              )
    )
)

particleFlowClusterEKUncorrected = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitShashlik"),
    recHitCleaners = cms.VPSet( ),
    seedFinder = _localMaxSeeds_EK,
    initialClusteringStep = _topoClusterizer_EK,
    pfClusterBuilder = _pfClusterizer_EK,
    positionReCalc =  cms.PSet( ),
    energyCorrector = cms.PSet( )
    )
