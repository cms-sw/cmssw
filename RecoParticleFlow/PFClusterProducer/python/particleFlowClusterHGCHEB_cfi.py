import FWCore.ParameterSet.Config as cms

#### PF CLUSTER HGCEE ####

#cleaning 

#seeding
_noseeds_HGCHEB = cms.PSet(
    algoName = cms.string("PassThruSeedFinder")   
)

#topo clusters
_arborClusterizer_HGCHEB = cms.PSet(
    algoName = cms.string("SimpleArborClusterizer"),    
    cellSize = cms.double(10.0),
    layerThickness = cms.double(7.0),
    thresholdsByDetector = cms.VPSet()
)

particleFlowClusterHGCHEB = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHGCHEB"),
    recHitCleaners = cms.VPSet(),
    seedFinder = _noseeds_HGCHEB,
    initialClusteringStep = _arborClusterizer_HGCHEB,
    pfClusterBuilder = cms.PSet( ),
    positionReCalc = cms.PSet(),
    energyCorrector = cms.PSet()
)

