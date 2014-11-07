import FWCore.ParameterSet.Config as cms

#### PF CLUSTER HGCEE ####

#cleaning 

#seeding
_localmaxseeds_HGCEE = cms.PSet(
    algoName = cms.string("LocalMaximumSeedFinder"),
    thresholdsByDetector = cms.VPSet(
    cms.PSet( detector = cms.string("HGC_ECAL"),
              seedingThreshold = cms.double(1e-6*1.75*55.1), #seeding threshold converted to GeV from keV
              seedingThresholdPt = cms.double(0.0)
              )
    ),
    nNeighbours = cms.int32(8)
)

#for arbor this is more a pre-clustering step to find little clusters
_arborTopoClusterizer_HGCEE = cms.PSet(
    algoName = cms.string("IntraLayerClusteringAlgorithm"),    
    IntraLayerMaxDistance = cms.double( 19.0 ), # hit separation in mm
    ShouldSplitClusterInSingleCaloHitClusters = cms.bool(False), # splitsmall clusters
    MaximumSizeForClusterSplitting = cms.uint32( 3 ), #largest of small clusters to split
    thresholdsByDetector = cms.VPSet( )
)
_simplePosCalcHGCEE =  cms.PSet(
    algoName = cms.string("SimplePositionCalc"),
    minFractionInCalc = cms.double(0.0)
)

#the real arbor clusterizer
_manqiArborClusterizer_HGCEE = cms.PSet(
    algoName = cms.string("SimpleArborClusterizer"), 
    # use basic pad sizes in HGCEE
    cellSize = cms.double(15.0),
    layerThickness = cms.double(25.0),
    distSeedForMerge = cms.double(20.0),
    killNoiseClusters = cms.bool(True),
    maxNoiseClusterSize = cms.uint32(3),
    allowSameLayerSeedMerge = cms.bool(False),
    thresholdsByDetector = cms.VPSet( )
)

_positionCalcHGCEE_onelayer = cms.PSet(
    algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
    ##
    minFractionInCalc = cms.double(1e-9),
    posCalcNCrystals = cms.int32(-1),
    logWeightDenominator = cms.double(1e-6*0.25*55.1), # use ADC value 0.25*MIP
    minAllowedNormalization = cms.double(1e-9)
    )

_fromScratchHGCClusterizer_HGCEE = cms.PSet(
    algoName = cms.string("HGCClusterizer"), 
    thresholdsByDetector = cms.VPSet( ),
    positionCalcInLayer = _positionCalcHGCEE_onelayer,
    moliereRadii = cms.PSet( HGC_ECAL = cms.double(2.9)  ) #cm
)

_arborClusterizer_HGCEE = cms.PSet(
    algoName = cms.string("ArborConnectorClusteringAlgorithm"), 
    # these are taken from the settings for Fine Granularity in ArborPFA
    MaximumForwardDistanceForConnection = cms.double(60.0), #in mm
    MaximumTransverseDistanceForConnection = cms.double(40.0), #in mm
    AllowForwardConnectionForIsolatedObjects = cms.bool(False),
    ShouldUseIsolatedObjects = cms.bool(True),
    MaximumNumberOfKeptConnectors = cms.uint32(5),
    OrderParameterAnglePower = cms.double(1.0),
    OrderParameterDistancePower = cms.double(0.5),
    minFractionToKeep = cms.double(1e-7)
)

#weights for layers from P.Silva (24 October 2014)
## this is for V5!
weight_vec = [0.080]
weight_vec.extend([0.62 for x in range(9)])
weight_vec.extend([0.81 for x in range(9)])
weight_vec.extend([1.19 for x in range(8)])

# MIP effective to 1.0/GeV (from fit to data of P. Silva)
#f(x) = a/(1-exp(-bx - c))
# x = cosh(eta)
# a = 82.8
# b = 1e6
# c = 1e6

_HGCEE_ElectronEnergy = cms.PSet(
    algoName = cms.string("HGCEEElectronEnergyCalibrator"),
    weights = cms.vdouble(weight_vec),
    effMip_to_InverseGeV_a = cms.double(82.8),
    effMip_to_InverseGeV_b = cms.double(1e6),
    effMip_to_InverseGeV_c = cms.double(1e6),
    MipValueInGeV = cms.double(55.1*1e-6)
)

particleFlowClusterHGCEE = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHGCEE"),
    recHitCleaners = cms.VPSet(),
    seedFinder = _localmaxseeds_HGCEE,
    initialClusteringStep = _fromScratchHGCClusterizer_HGCEE,
    pfClusterBuilder = cms.PSet( ), #_arborClusterizer_HGCEE,
    positionReCalc = cms.PSet( ), #_simplePosCalcHGCEE,
    energyCorrector = _HGCEE_ElectronEnergy
)

