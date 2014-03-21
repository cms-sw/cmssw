import FWCore.ParameterSet.Config as cms

hltL2TauIsolationProducer = cms.EDProducer( "L2TauIsolationProducer",
    L2TauJetCollection = cms.InputTag( "hltL2TauJets" ),
    EBRecHits = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEE' ),
    crystalThreshold = cms.double( 0.1 ),
    towerThreshold = cms.double(1.0 ),
    ECALIsolation = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      innerCone = cms.double( 0.15 ),
      outerCone = cms.double( 0.5 )
    ),
    ECALClustering = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      clusterRadius = cms.double( 0.08 )
    ),
    TowerIsolation = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      innerCone = cms.double( 0.2 ),
      outerCone = cms.double( 0.5 )
    )
)



