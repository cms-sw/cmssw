import FWCore.ParameterSet.Config as cms

L2TauIsolationProducer = cms.EDProducer("L2TauIsolationProducer",
    #Configure ECAL Isolation		
    ECALIsolation = cms.PSet(
        innerCone = cms.double(0.15),
        runAlgorithm = cms.bool(True),
        outerCone = cms.double(0.5)
    ),
    TowerIsolation = cms.PSet(
        innerCone = cms.double(0.2),
        runAlgorithm = cms.bool(True),
        outerCone = cms.double(0.5)
    ),
    EERecHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    EBRecHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    #pick the collection you want to access	 
    L2TauJetCollection = cms.InputTag("l2TauJetsProvider","DoubleTau"),
    #Configure ECAL Clustering
    ECALClustering = cms.PSet(
        runAlgorithm = cms.bool(True),
        clusterRadius = cms.double(0.08) ##Radius for Clusters

    ),
    towerThreshold = cms.double(0.2),
    crystalThreshold = cms.double(0.1)
)


