import FWCore.ParameterSet.Config as cms

hltL2TauIsolationSelector = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag('hltL2TauIsolationProducer'),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)



