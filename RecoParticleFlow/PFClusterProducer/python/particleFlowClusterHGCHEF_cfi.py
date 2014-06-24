import FWCore.ParameterSet.Config as cms

#### PF CLUSTER HGCEF ####

#cleaning 

#seeding
_noseeds_HGCHEF = cms.PSet(
    algoName = cms.string("PassThruSeedFinder")   
)

#topo clusters
_arborClusterizer_HGCHEF = cms.PSet(
    algoName = cms.string("SimpleArborClusterizer"),    
    cellSize = cms.double(3.0),
    layerThickness = cms.double(2.5),
    thresholdsByDetector = cms.VPSet()
)

particleFlowClusterHGCHEF = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHGCHEF"),
    recHitCleaners = cms.VPSet(),
    seedFinder = _noseeds_HGCHEF,
    initialClusteringStep = _arborClusterizer_HGCHEF,
    pfClusterBuilder = cms.PSet( ),
    positionReCalc = cms.PSet(),
    energyCorrector = cms.PSet()
)

