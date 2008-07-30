import FWCore.ParameterSet.Config as cms

#Muon Tau Path 
MuonTauDQMPFTaus = cms.EDFilter("HLTTauDQMOfflineSource",
                            DQMFolder                   = cms.string('HLTOffline/HLTTAU/MuonTau/'),
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
                            MainFilter                  = cms.InputTag("hltFilterIsolatedTauJetsL25MuonTau","","HLT"),
                            L1BackupFilter              = cms.InputTag("NONE"),
                            L2BackupFilter              = cms.InputTag("NONE"),
                            L25BackupFilter             = cms.InputTag("NONE"),
                            L3BackupFilter              = cms.InputTag("NONE"),
                            refTauObjects               = cms.InputTag("TauRefProducer","PFTaus"),
                            refLeptonObjects            = cms.InputTag("TauRefProducer","Muons"),
                            matchingDeltaR              = cms.double(0.3),
                            HistEtMin                   = cms.double(0.),
                            HistEtMax                   = cms.double(100),
                            HistNEtBins                 = cms.int32(50),
                            HistNEtaBins                = cms.int32(25)
                            )

#-----------L2 Monitoring Ref PFTaus
MuonTauDQML2RefPFTaus = cms.EDFilter("HLTTauCaloDQMOfflineSource",
                                  DQMFolder              = cms.string('HLTOffline/HLTTAU/MuonTau/L2_Matched_To_PFTau'),
                                  L2InfoAssociationInput = cms.InputTag("hltL2MuonTauIsolationProducer","L2TauIsolationInfoAssociator"),
                                  refCollection          = cms.InputTag("TauRefProducer","PFTaus"),
                                  MET                    = cms.InputTag("hltMet"),
                                  doReference            = cms.bool(True),
                                  MatchDeltaR            = cms.double(0.3),
                                  OutputFileName         = cms.string(''),
                                  L2IsolatedJets         = cms.InputTag("hltL2MuonTauIsolationSelector","Isolated"),
                                  EtMin                  = cms.double(0.),
                                  EtMax                  = cms.double(100.),
                                  NBins                  = cms.int32(50)
                              )

#----------L2Monitoring Ref CaloTaus
MuonTauDQML2RefCaloTaus = cms.EDFilter("HLTTauCaloDQMOfflineSource",
                                  DQMFolder              = cms.string('HLTOffline/HLTTAU/MuonTau/L2_Matched_To_CaloTau'),
                                  L2InfoAssociationInput = cms.InputTag("hltL2MuonTauIsolationProducer","L2TauIsolationInfoAssociator"),
                                  refCollection          = cms.InputTag("TauRefProducer","CaloTaus"),
                                  MET                    = cms.InputTag("hltMet"),
                                  doReference            = cms.bool(True),
                                  MatchDeltaR            = cms.double(0.3),
                                  OutputFileName         = cms.string(''),
                                  L2IsolatedJets         = cms.InputTag("hltL2MuonTauIsolationSelector","Isolated"),
                                  EtMin                  = cms.double(0.),
                                  EtMax                  = cms.double(100.),
                                  NBins                  = cms.int32(50)
                              )
#---------L2 Monitoring Ref Electrons
MuonTauDQML2RefElectrons = cms.EDFilter("HLTTauCaloDQMOfflineSource",
                                  DQMFolder              = cms.string('HLTOffline/HLTTAU/MuonTau/L2_Matched_To_Muons'),
                                  L2InfoAssociationInput = cms.InputTag("hltL2MuonTauIsolationProducer","L2TauIsolationInfoAssociator"),
                                  refCollection          = cms.InputTag("TauRefProducer","Electrons"),
                                  MET                    = cms.InputTag("hltMet"),
                                  doReference            = cms.bool(True),
                                  MatchDeltaR            = cms.double(0.3),
                                  OutputFileName         = cms.string(''),
                                  L2IsolatedJets         = cms.InputTag("hltL2MuonTauIsolationSelector","Isolated"),
                                  EtMin                  = cms.double(0.),
                                  EtMax                  = cms.double(100.),
                                  NBins                  = cms.int32(50)
                              )





#---------L25Monitoring Ref PFTaus
MuonTauDQML25RefPFTaus = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/MuonTau/L25_Matched_To_PFtaus'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL25MuonTau"),
                                   InputJets              = cms.InputTag("hltL2MuonTauIsolationSelector","Isolated"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL25MuonTau"),
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
MuonTauDQML25RefCaloTaus = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/MuonTau/L25_Matched_To_Calotaus'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL25MuonTau"),
                                   InputJets              = cms.InputTag("hltL2MuonTauIsolationSelector","Isolated"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL25MuonTau"),
                                   refCollection          = cms.InputTag("TauRefProducer","CaloTaus"),
                                   Type                   = cms.string('L25'),
                                   doReference            = cms.bool(True),
                                   MatchDeltaR            = cms.double(0.3),
                                   OutputFileName         = cms.string(''),
                                   EtMin                  = cms.double(0.),
                                   EtMax                  = cms.double(100.),
                                   NBins                  = cms.int32(50)
                               )

#---------L25Monitoring Ref Muons
MuonTauDQML25RefElectrons = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/MuonTau/L25_Matched_To_Muons'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL25MuonTau"),
                                   InputJets              = cms.InputTag("hltL2MuonTauIsolationSelector","Isolated"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL25MuonTau"),
                                   refCollection          = cms.InputTag("TauRefProducer","Electrons"),
                                   Type                   = cms.string('L25'),
                                   doReference            = cms.bool(True),
                                   MatchDeltaR            = cms.double(0.3),
                                   OutputFileName         = cms.string('out.root'),
                                   EtMin                  = cms.double(0.),
                                   EtMax                  = cms.double(100.),
                                   NBins                  = cms.int32(50)
                               )

HLTTauOfflineDQMMuonTauSequence = cms.Sequence(MuonTauDQMPFTaus+MuonTauDQML2RefPFTaus+MuonTauDQML2RefCaloTaus+MuonTauDQML2RefElectrons+MuonTauDQML25RefPFTaus+MuonTauDQML25RefCaloTaus+MuonTauDQML25RefElectrons)







