import FWCore.ParameterSet.Config as cms

#### PF CLUSTER HGCEF ####

#cleaning 

#seeding
_noseeds_HGCHEF = cms.PSet(
    algoName = cms.string("PassThruSeedFinder")   
)

#topo clusters
#for arbor this is more a pre-clustering step to find little clusters
_arborTopoClusterizer_HGCHEF = cms.PSet(
    algoName = cms.string("IntraLayerClusteringAlgorithm"),    
    IntraLayerMaxDistance = cms.double( 96.0 ), # hit separation in mm
    ShouldSplitClusterInSingleCaloHitClusters = cms.bool(False), # splitsmall clusters
    MaximumSizeForClusterSplitting = cms.uint32( 3 ), #largest of small clusters to split
    thresholdsByDetector = cms.VPSet( )
)

particleFlowClusterHGCHEF = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHGCHEF"),
    recHitCleaners = cms.VPSet(),
    seedFinder = _noseeds_HGCHEF,
    initialClusteringStep = _arborTopoClusterizer_HGCHEF,
    pfClusterBuilder = cms.PSet( ),
    positionReCalc = cms.PSet(),
    energyCorrector = cms.PSet()
)

