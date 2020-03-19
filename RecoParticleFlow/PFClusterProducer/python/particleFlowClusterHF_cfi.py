import FWCore.ParameterSet.Config as cms

#### PF CLUSTER HFEM ####

#cleaning
_spikeAndDoubleSpikeCleaner_HF = cms.PSet(
    algoName = cms.string("SpikeAndDoubleSpikeCleaner"),    
    cleaningByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("HF_EM"),
                 #single spike
                 singleSpikeThresh = cms.double(80.0),
                 minS4S1_a = cms.double(0.11), #constant term
                 minS4S1_b = cms.double(-0.19), #log pt scaling
                 #double spike
                 doubleSpikeThresh = cms.double(1e9),
                 doubleSpikeS6S2 = cms.double(-1.0),
                 energyThresholdModifier = cms.double(1.0), ## aka "tighterE"
                 fractionThresholdModifier = cms.double(1.0) ## aka "tighterF"
                 ),
       cms.PSet( detector = cms.string("HF_HAD"),
                 #single spike
                 singleSpikeThresh = cms.double(120.0),
                 minS4S1_a = cms.double(0.045), #constant term
                 minS4S1_b = cms.double(-0.080), #log pt scaling
                 #double spike
                 doubleSpikeThresh = cms.double(1e9),
                 doubleSpikeS6S2 = cms.double(-1.0),
                 energyThresholdModifier = cms.double(1.0), ## aka "tighterE"
                 fractionThresholdModifier = cms.double(1.0) ## aka "tighterF"
              )
       )
    )

#seeding
_localMaxSeeds_HF = cms.PSet(
    algoName = cms.string("LocalMaximumSeedFinder"),
    thresholdsByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("HF_EM"),
              seedingThreshold = cms.double(1.4),
              seedingThresholdPt = cms.double(0.0)
              ),
    cms.PSet( detector = cms.string("HF_HAD"),
              seedingThreshold = cms.double(1.4),
              seedingThresholdPt = cms.double(0.0)
              )

    ),
    nNeighbours = cms.int32(0)
)

#topo clusters
_topoClusterizer_HF = cms.PSet(
    algoName = cms.string("Basic2DGenericTopoClusterizer"),
    thresholdsByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("HF_EM"),
              gatheringThreshold = cms.double(0.8),
              gatheringThresholdPt = cms.double(0.0)
              ),
    cms.PSet( detector = cms.string("HF_HAD"),
              gatheringThreshold = cms.double(0.8),
              gatheringThresholdPt = cms.double(0.0)
              )

    ),
    useCornerCells = cms.bool(False)
)

#position calc
_positionCalcHF_cross_nodepth = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
    ##
    minFractionInCalc = cms.double(1e-9),
    posCalcNCrystals = cms.int32(5),
    logWeightDenominator = cms.double(0.8), # same as gathering threshold
    minAllowedNormalization = cms.double(1e-9)    
)

_positionCalcHF_all_nodepth = _positionCalcHF_cross_nodepth.clone(
    posCalcNCrystals = cms.int32(-1)
    )

#pf clusters
_pfClusterizer_HF = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowClusterizer"),
    #pf clustering parameters
    minFractionToKeep = cms.double(1e-7),
    positionCalc = _positionCalcHF_cross_nodepth,
    allCellsPositionCalc = _positionCalcHF_all_nodepth,
    showerSigma = cms.double(10.0),
    stoppingTolerance = cms.double(1e-8),
    maxIterations = cms.uint32(50),
    excludeOtherSeeds = cms.bool(True),
    minFracTot = cms.double(1e-20), ## numerical stabilization
    recHitEnergyNorms = cms.VPSet(
    cms.PSet( detector = cms.string("HF_EM"),
              recHitEnergyNorm = cms.double(0.8)
              ),
    cms.PSet( detector = cms.string("HF_HAD"),
              recHitEnergyNorm = cms.double(0.8)
              )
    )
)

particleFlowClusterHF = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHF"),
    recHitCleaners = cms.VPSet(),
    seedCleaners = cms.VPSet(),
    seedFinder = _localMaxSeeds_HF,
    initialClusteringStep = _topoClusterizer_HF,
    pfClusterBuilder = _pfClusterizer_HF,
    positionReCalc = cms.PSet(),
    energyCorrector = cms.PSet()
    )

