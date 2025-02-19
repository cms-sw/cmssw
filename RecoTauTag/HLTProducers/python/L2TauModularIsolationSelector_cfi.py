import FWCore.ParameterSet.Config as cms


hltL2TauRelaxingIsolationSelector = cms.EDProducer( "L2TauModularIsolationSelector",
    L2InfoAssociation = cms.InputTag( "hltL2TauIsolationProducer" ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 ),
    EcalIsolationEt = cms.vdouble( 5.0, 0.025, 7.5E-4 ),
    NumberOfECALClusters = cms.vdouble( 1000.0, 0.0, 0.0 ),
    ECALClusterPhiRMS = cms.vdouble( 1000.0, 0.0, 0.0 ),
    ECALClusterEtaRMS = cms.vdouble( 1000.0, 0.0, 0.0 ),
    ECALClusterDRRMS = cms.vdouble( 1000.0, 0.0, 0.0 ),
    HcalIsolationEt = cms.vdouble( 100.0, 0.0, 0.0 ),
    NumberOfHCALClusters = cms.vdouble( 1000.0, 0.0, 0.0 ),
    HCALClusterPhiRMS = cms.vdouble( 1000.0, 0.0, 0.0 ),
    HCALClusterEtaRMS = cms.vdouble( 1000.0, 0.0, 0.0 ),
    HCALClusterDRRMS = cms.vdouble( 1000.0, 0.0, 0.0 )                                                    
)

