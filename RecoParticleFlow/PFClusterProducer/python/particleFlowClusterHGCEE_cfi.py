import FWCore.ParameterSet.Config as cms

#### PF CLUSTER HGCEE ####

#cleaning 

#seeding
_noseeds_HGCEE = cms.PSet(
    algoName = cms.string("PassThruSeedFinder")   
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

particleFlowClusterHGCEE = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHGCEE"),
    recHitCleaners = cms.VPSet(),
    seedFinder = _noseeds_HGCEE,
    initialClusteringStep = _arborTopoClusterizer_HGCEE,
    pfClusterBuilder = cms.PSet( ), #_arborClusterizer_HGCEE,
    positionReCalc = cms.PSet( ), #_simplePosCalcHGCEE,
    energyCorrector = cms.PSet()
)

