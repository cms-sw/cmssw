import FWCore.ParameterSet.Config as cms

spikeAndDoubleSpikeCleaner_EB = cms.PSet(
    algoName = cms.string("SpikeAndDoubleSpikeCleaner"),
    #single spike
    cleaningThreshold = cms.double(4.0),
    minS4S1_a = cms.double(0.04), #constant term
    minS4S1_b = cms.double(-0.024), #log pt scaling
    #double spike
    doubleSpikeThresh = cms.double(10.0),
    doubleSpikeS6S2 = cms.double(0.04),
    energyThresholdModifier = cms.double(2.0), ## aka "tighterE"
    fractionThresholdModifier = cms.double(3.0) ## aka "tighterF"
    )

spikeAndDoubleSpikeCleaner_EE = cms.PSet(
    algoName = cms.string("SpikeAndDoubleSpikeCleaner"),
    #single spike
    cleaningThreshold = cms.double(15.0),
    minS4S1_a = cms.double(0.02), #constant term
    minS4S1_b = cms.double(-0.0125), #log pt scaling
    #double spike
    doubleSpikeThresh = cms.double(1e9),
    doubleSpikeS6S2 = cms.double(-1.0),
    energyThresholdModifier = cms.double(2.0), ## aka "tighterE"
    fractionThresholdModifier = cms.double(3.0) ## aka "tighterF"
    )

localMaxSeeds_EB = cms.PSet(
    algoName = cms.string("LocalMaximum2DSeedFinder"),
    ### seed finding parameters
    seedingThreshold = cms.double(0.23),
    seedingThresholdPt = cms.double(0.0),
    nNeighbours = cms.uint32(8)
    )

localMaxSeeds_EE = localMaxSeeds_EB.clone(
    seedingThreshold = cms.double(0.6),
    seedingThresholdPt = cms.double(0.15)
    )

topoClusterizer_EB = cms.PSet(
    algoName = cms.string("Basic2DGenericTopoClusterizer"),
    #topo clustering parameters
    gatheringThreshold = cms.double(0.08),
    gatheringThresholdPt = cms.double(0.0),
    useCornerCells = cms.bool(True)
    )

topoClusterizer_EE = topoClusterizer_EB.clone(
    gatheringThreshold = cms.double(0.3),
    gatheringThresholdPt = cms.double(0.0)
    )

positionCalcEB_all_nodepth = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
    ##
    minFractionInCalc = cms.double(1e-9),
    posCalcNCrystals = cms.int32(-1),
    logWeightDenominator = cms.double(0.08), # same as gathering threshold
    minAllowedNormalization = cms.double(1e-9)
    )

positionCalcEE_all_nodepth = positionCalcEB_all_nodepth.clone(
    #in the old PFClusterAlgo this is same as barrel
    # logWeightDenominator = cms.double(0.3) 
    )

positionCalcEB_3x3_nodepth = positionCalcEB_all_nodepth.clone(
    posCalcNCrystals = cms.int32(9)
    )

positionCalcEE_3x3_nodepth = positionCalcEE_all_nodepth.clone(
    posCalcNCrystals = cms.int32(9)
    )

positionCalcECAL_all_withdepth = cms.PSet(
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

pfClusterizer_EB = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowClusterizer"),
    #pf clustering parameters
    minFractionToKeep = cms.double(1e-7),
    positionCalc = positionCalcEB_3x3_nodepth,
    allCellsPositionCalc = positionCalcEB_all_nodepth,
    positionCalcForConvergence = positionCalcECAL_all_withdepth,
    showerSigma = cms.double(1.5),
    stoppingTolerance = cms.double(1e-8),
    maxIterations = cms.uint32(50),
    excludeOtherSeeds = cms.bool(True)
    )

pfClusterizer_EE = pfClusterizer_EB.clone(   
    #positionCalc = positionCalcEE_3x3_nodepth,
    #allCellsPositionCalc = positionCalcEE_all_nodepth    
    )

particleFlowClusterECALBarrel = cms.EDProducer(
    "PFClusterProducerNew",
    recHitsSource = cms.InputTag("particleFlowRecHitECAL:EB"),
    recHitCleaners = cms.VPSet(spikeAndDoubleSpikeCleaner_EB),
    seedFinder = localMaxSeeds_EB,
    topoClusterBuilder = topoClusterizer_EB,
    pfClusterBuilder = pfClusterizer_EB,
    positionReCalc = positionCalcECAL_all_withdepth
    )

particleFlowClusterECALEndcap = cms.EDProducer(
    "PFClusterProducerNew",
    recHitsSource = cms.InputTag("particleFlowRecHitECAL:EE"),
    recHitCleaners = cms.VPSet(spikeAndDoubleSpikeCleaner_EE),
    seedFinder = localMaxSeeds_EE,
    topoClusterBuilder = topoClusterizer_EE,
    pfClusterBuilder = pfClusterizer_EE,
    positionReCalc = positionCalcECAL_all_withdepth
    )

particleFlowClusterECALNew = cms.EDProducer(
    "PFClusterCollectionMerger",
    inputs = cms.VInputTag(cms.InputTag("particleFlowClusterECALBarrel"),
                           cms.InputTag("particleFlowClusterECALEndcap"))
    )
