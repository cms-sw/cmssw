import FWCore.ParameterSet.Config as cms

#### PF CLUSTER HGCEE ####

#cleaning 

#seeding
_noseeds_HGCEE = cms.PSet(
    algoName = cms.string("PassThruSeedFinder")   
)

#topo clusters
_arborClusterizer_HGCEE = cms.PSet(
    algoName = cms.string("SimpleArborClusterizer"),    
    cellSize = cms.double(2.0),
    layerThickness = cms.double(2.0),
    thresholdsByDetector = cms.VPSet()
)

particleFlowClusterHGCEE = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHGCEE"),
    recHitCleaners = cms.VPSet(),
    seedFinder = _noseeds_HGCEE,
    initialClusteringStep = _arborClusterizer_HGCEE,
    pfClusterBuilder = cms.PSet( ),
    positionReCalc = cms.PSet(),
    energyCorrector = cms.PSet()
)

