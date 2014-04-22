import FWCore.ParameterSet.Config as cms

from particleFlowClusterECALTimeResolutionParameters_cfi import _timeResolutionECALBarrel, _timeResolutionECALEndcap

#### PF CLUSTER ECAL ####

#cleaning
_spikeAndDoubleSpikeCleaner_ECAL = cms.PSet(
    algoName = cms.string("SpikeAndDoubleSpikeCleaner"),    
    cleaningByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("ECAL_BARREL"),
              #single spike
              singleSpikeThresh = cms.double(4.0),
              minS4S1_a = cms.double(0.04), #constant term
              minS4S1_b = cms.double(-0.024), #log pt scaling
              #double spike
              doubleSpikeThresh = cms.double(10.0),
              doubleSpikeS6S2 = cms.double(0.04),
              energyThresholdModifier = cms.double(2.0), ## aka "tighterE"
              fractionThresholdModifier = cms.double(3.0) ## aka "tighterF"
              ),
    cms.PSet( detector = cms.string("ECAL_ENDCAP"),
              #single spike
              singleSpikeThresh = cms.double(15.0),
              minS4S1_a = cms.double(0.02), #constant term
              minS4S1_b = cms.double(-0.0125), #log pt scaling
              #double spike
              doubleSpikeThresh = cms.double(1e9),
              doubleSpikeS6S2 = cms.double(-1.0),
              energyThresholdModifier = cms.double(2.0), ## aka "tighterE"
              fractionThresholdModifier = cms.double(3.0) ## aka "tighterF"
              )
    )
)

#seeding
_localMaxSeeds_ECAL = cms.PSet(
    algoName = cms.string("LocalMaximumSeedFinder"),
    thresholdsByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("ECAL_ENDCAP"),
              seedingThreshold = cms.double(0.6),
              seedingThresholdPt = cms.double(0.15)
              ),
    cms.PSet( detector = cms.string("ECAL_BARREL"),
              seedingThreshold = cms.double(0.23),
              seedingThresholdPt = cms.double(0.0)
              )
    ),
    nNeighbours = cms.int32(8)
)

# topo clusterizer
_topoClusterizer_ECAL = cms.PSet(
    algoName = cms.string("Basic2DGenericTopoClusterizer"),
    thresholdsByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("ECAL_BARREL"),
              gatheringThreshold = cms.double(0.08),
              gatheringThresholdPt = cms.double(0.0)
              ),
    cms.PSet( detector = cms.string("ECAL_ENDCAP"),
              gatheringThreshold = cms.double(0.3),
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
    timeResolutionCalcBarrel = _timeResolutionECALBarrel,
    timeResolutionCalcEndcap = _timeResolutionECALEndcap,
)
_positionCalcECAL_3x3_nodepth = _positionCalcECAL_all_nodepth.clone(
    posCalcNCrystals = cms.int32(9)
)
_positionCalcECAL_all_withdepth = cms.PSet(
    algoName = cms.string("ECAL2DPositionCalcWithDepthCorr"),
    ##
    minFractionInCalc = cms.double(0.0),
    minAllowedNormalization = cms.double(0.0),
    T0_EB = cms.double(7.4),
    T0_EE = cms.double(3.1),
    T0_ES = cms.double(1.2),
    W0 = cms.double(4.2),
    X0 = cms.double(0.89)
)

# pf clustering
_pfClusterizer_ECAL = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowClusterizer"),
    #pf clustering parameters
    minFractionToKeep = cms.double(1e-7),
    positionCalc = _positionCalcECAL_3x3_nodepth,
    allCellsPositionCalc = _positionCalcECAL_all_nodepth,
    positionCalcForConvergence = _positionCalcECAL_all_withdepth,
    showerSigma = cms.double(1.5),
    stoppingTolerance = cms.double(1e-8),
    maxIterations = cms.uint32(50),
    excludeOtherSeeds = cms.bool(True),
    minFracTot = cms.double(1e-20), ## numerical stabilization
    recHitEnergyNorms = cms.VPSet(
    cms.PSet( detector = cms.string("ECAL_BARREL"),
              recHitEnergyNorm = cms.double(0.08)
              ),
    cms.PSet( detector = cms.string("ECAL_ENDCAP"),
              recHitEnergyNorm = cms.double(0.3)
              )
    )
)

particleFlowClusterECALUncorrected = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitECAL"),
    recHitCleaners = cms.VPSet(_spikeAndDoubleSpikeCleaner_ECAL),
    seedFinder = _localMaxSeeds_ECAL,
    initialClusteringStep = _topoClusterizer_ECAL,
    pfClusterBuilder = _pfClusterizer_ECAL,
    positionReCalc = _positionCalcECAL_all_withdepth,
    energyCorrector = cms.PSet()
    )
