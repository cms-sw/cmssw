import FWCore.ParameterSet.Config as cms


hltL2TauRelaxingIsolationSelector = cms.EDProducer( "L2TauRelaxingIsolationSelector",
                                                L2InfoAssociation = cms.InputTag( 'hltL2TauIsolationProducer'),
                                                EcalIsolationEt = cms.vdouble( 5.0,0.025,0.0),
                                                TowerIsolationEt = cms.vdouble( 1000.0,0.0,0.0 ),
                                                ClusterEtaRMS = cms.vdouble( 1000.0,0.0,0.0),
                                                ClusterPhiRMS = cms.vdouble( 1000.0,0.0,0.0),
                                                ClusterDRRMS = cms.vdouble( 1000,0.0,0.0),
                                                NumberOfClusters=cms.vdouble( 1000.0,0.0,0.0),
                                                MinJetEt = cms.double( 0.0),
                                                SeedTowerEt = cms.double( -10.0 )
                                            )
