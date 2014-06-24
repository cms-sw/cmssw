import FWCore.ParameterSet.Config as cms

#### PF CLUSTER HGCEE ####

#cleaning 

#seeding
_noseeds_HGCHEB = cms.PSet(
    algoName = cms.string("PassThruSeedFinder")   
)

#topo clusters
#for arbor this is more a pre-clustering step to find little clusters
_arborTopoClusterizer_HGCHEB = cms.PSet(
    algoName = cms.string("IntraLayerClusteringAlgorithm"),    
    IntraLayerMaxDistance = cms.double( 186.0 ), # hit separation in mm
    ShouldSplitClusterInSingleCaloHitClusters = cms.bool(False), # splitsmall clusters
    MaximumSizeForClusterSplitting = cms.uint32( 3 ), #largest of small clusters to split
    thresholdsByDetector = cms.VPSet( )
)

particleFlowClusterHGCHEB = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHGCHEB"),
    recHitCleaners = cms.VPSet(),
    seedFinder = _noseeds_HGCHEB,
    initialClusteringStep = _arborTopoClusterizer_HGCHEB,
    pfClusterBuilder = cms.PSet( ),
    positionReCalc = cms.PSet(),
    energyCorrector = cms.PSet()
)

