import FWCore.ParameterSet.Config as cms

#Electron Tau Path 
ElectronTauDQMPFTaus = cms.EDFilter("HLTTauDQMOfflineSource",
                            DQMFolder                   = cms.string('HLTOffline/HLTTAU/ElectronTau/'),
                            monitorName                 = cms.string('RefToPFTaus'),
                            outputFile                  = cms.string(''),
                            prescaleEvt                 = cms.int32(-1),
                            disableROOToutput           = cms.bool(True),
                            verbose                     = cms.bool(False),
                            NTriggeredTaus              = cms.uint32(1),
                            NTriggeredLeptons           = cms.uint32(1),
                            LeptonPDGId                 = cms.int32(11),
                            TauPDGId                    = cms.int32(15),
                            TriggerEvent                = cms.InputTag("hltTriggerSummaryAOD","",""),
                            MainFilter                  = cms.InputTag("hltFilterIsolatedTauJetsL25ElectronTau","","HLT"),
                            L1BackupFilter              = cms.InputTag("NONE"),
                            L2BackupFilter              = cms.InputTag("NONE"),
                            L25BackupFilter             = cms.InputTag("NONE"),
                            L3BackupFilter              = cms.InputTag("NONE"),
                            refTauObjects               = cms.InputTag("TauRefProducer","PFTaus"),
                            refLeptonObjects            = cms.InputTag("TauRefProducer","Electrons"),
                            matchingDeltaR              = cms.double(0.3),
                            HistEtMin                   = cms.double(0.),
                            HistEtMax                   = cms.double(100),
                            HistNEtBins                 = cms.int32(50),
                            HistNEtaBins                = cms.int32(25)
                            )

#-----------L2 Monitoring Ref PFTaus
ElectronTauDQML2RefPFTaus = cms.EDFilter("HLTTauCaloDQMOfflineSource",
                                  DQMFolder              = cms.string('HLTOffline/HLTTAU/ElectronTau/L2_Matched_To_PFTau'),
                                  L2InfoAssociationInput = cms.InputTag("hltL2ElectronTauIsolationProducer","L2TauIsolationInfoAssociator"),
                                  refCollection          = cms.InputTag("TauRefProducer","PFTaus"),
                                  MET                    = cms.InputTag("hltMet"),
                                  doReference            = cms.bool(True),
                                  MatchDeltaR            = cms.double(0.3),
                                  OutputFileName         = cms.string(''),
                                  L2IsolatedJets         = cms.InputTag("hltL2ElectronTauIsolationSelector","Isolated"),
                                  EtMin                  = cms.double(0.),
                                  EtMax                  = cms.double(100.),
                                  NBins                  = cms.int32(50)
                              )

#----------L2Monitoring Ref CaloTaus
ElectronTauDQML2RefCaloTaus = cms.EDFilter("HLTTauCaloDQMOfflineSource",
                                  DQMFolder              = cms.string('HLTOffline/HLTTAU/ElectronTau/L2_Matched_To_CaloTau'),
                                  L2InfoAssociationInput = cms.InputTag("hltL2ElectronTauIsolationProducer","L2TauIsolationInfoAssociator"),
                                  refCollection          = cms.InputTag("TauRefProducer","CaloTaus"),
                                  MET                    = cms.InputTag("hltMet"),
                                  doReference            = cms.bool(True),
                                  MatchDeltaR            = cms.double(0.3),
                                  OutputFileName         = cms.string(''),
                                  L2IsolatedJets         = cms.InputTag("hltL2ElectronTauIsolationSelector","Isolated"),
                                  EtMin                  = cms.double(0.),
                                  EtMax                  = cms.double(100.),
                                  NBins                  = cms.int32(50)
                              )
#---------L2 Monitoring Ref Electrons
ElectronTauDQML2RefElectrons = cms.EDFilter("HLTTauCaloDQMOfflineSource",
                                  DQMFolder              = cms.string('HLTOffline/HLTTAU/ElectronTau/L2_Matched_To_Electrons'),
                                  L2InfoAssociationInput = cms.InputTag("hltL2ElectronTauIsolationProducer","L2TauIsolationInfoAssociator"),
                                  refCollection          = cms.InputTag("TauRefProducer","Electrons"),
                                  MET                    = cms.InputTag("hltMet"),
                                  doReference            = cms.bool(True),
                                  MatchDeltaR            = cms.double(0.3),
                                  OutputFileName         = cms.string(''),
                                  L2IsolatedJets         = cms.InputTag("hltL2ElectronTauIsolationSelector","Isolated"),
                                  EtMin                  = cms.double(0.),
                                  EtMax                  = cms.double(100.),
                                  NBins                  = cms.int32(50)
                              )





#---------L25Monitoring Ref PFTaus
ElectronTauDQML25RefPFTaus = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/ElectronTau/L25_Matched_To_PFtaus'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL25ElectronTau"),
                                   InputJets              = cms.InputTag("hltL2ElectronTauIsolationSelector","Isolated"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL25ElectronTau"),
                                   refCollection          = cms.InputTag("TauRefProducer","PFTaus"),
                                   Type                   = cms.string('L25'),
                                   doReference            = cms.bool(True),
                                   MatchDeltaR            = cms.double(0.3),
                                   OutputFileName         = cms.string(''),
                                   EtMin                  = cms.double(0.),
                                   EtMax                  = cms.double(100.),
                                   NBins                  = cms.int32(50)
                               )

#---------L25Monitoring Ref CaloTaus
ElectronTauDQML25RefCaloTaus = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/ElectronTau/L25_Matched_To_Calotaus'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL25ElectronTau"),
                                   InputJets              = cms.InputTag("hltL2ElectronTauIsolationSelector","Isolated"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL25ElectronTau"),
                                   refCollection          = cms.InputTag("TauRefProducer","CaloTaus"),
                                   Type                   = cms.string('L25'),
                                   doReference            = cms.bool(True),
                                   MatchDeltaR            = cms.double(0.3),
                                   OutputFileName         = cms.string(''),
                                   EtMin                  = cms.double(0.),
                                   EtMax                  = cms.double(100.),
                                   NBins                  = cms.int32(50)
                               )

#---------L25Monitoring Ref Electrons
ElectronTauDQML25RefElectrons = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/ElectronTau/L25_Matched_To_Electrons'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL25ElectronTau"),
                                   InputJets              = cms.InputTag("hltL2ElectronTauIsolationSelector","Isolated"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL25ElectronTau"),
                                   refCollection          = cms.InputTag("TauRefProducer","Electrons"),
                                   Type                   = cms.string('L25'),
                                   doReference            = cms.bool(True),
                                   MatchDeltaR            = cms.double(0.3),
                                   OutputFileName         = cms.string(''),
                                   EtMin                  = cms.double(0.),
                                   EtMax                  = cms.double(100.),
                                   NBins                  = cms.int32(50)
                               )

HLTTauOfflineDQMElectronTauSequence = cms.Sequence(ElectronTauDQMPFTaus+ElectronTauDQML2RefPFTaus+ElectronTauDQML2RefCaloTaus+ElectronTauDQML2RefElectrons+ElectronTauDQML25RefPFTaus+ElectronTauDQML25RefCaloTaus+ElectronTauDQML25RefElectrons)







