import FWCore.ParameterSet.Config as cms


hltL2TauIsolationProducer = cms.EDProducer( "L2TauModularIsolationProducer",
    L2TauJetCollection = cms.InputTag( "hltL2TauJets" ),
    EBRecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEE' ),
    CaloTowers = cms.InputTag('hltTowerMakerForTaus'),

    pfClustersECAL = cms.InputTag('hltParticleFlowClusterECAL'),                                        
    pfClustersHCAL = cms.InputTag('hltParticleFlowClusterHCAL'),

    ecalIsolationAlgorithm = cms.string('recHits'),
    hcalIsolationAlgorithm = cms.string('recHits'),
    ecalClusteringAlgorithm = cms.string('particleFlow'),
    hcalClusteringAlgorithm = cms.string('particleFlow'),

    associationRadius = cms.double(0.5 ),

    #For simple Clustering Algorithm 
    simpleClusterRadiusEcal = cms.double(0.1 ),
    simpleClusterRadiusHcal = cms.double(0.18 ),
                                            

    #isolation Cones for Everyhting
    innerConeECAL = cms.double( 0.15 ),
    outerConeECAL = cms.double( 0.5 ),
    innerConeHCAL = cms.double( 0.25 ),
    outerConeHCAL = cms.double( 0.5 ),

   #You need those seriously if you plan to not use PF/bus recHits/CLusters in any algorithm
    crystalThresholdEE = cms.double( 0.45 ),
    crystalThresholdEB = cms.double( 0.12 ),
    towerThreshold = cms.double(0.1 )                                        
)

