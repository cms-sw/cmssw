import FWCore.ParameterSet.Config as cms

#Single Tau Path 
SingleTauDQMPFTaus = cms.EDFilter("HLTTauDQMOfflineSource",
                            DQMFolder                   = cms.string('HLTOffline/HLTTAU/SingleTau/'),
                            monitorName                 = cms.string('RefToPFTaus'),
                            outputFile                  = cms.string(''),
                            prescaleEvt                 = cms.int32(-1),
                            disableROOToutput           = cms.bool(True),
                            verbose                     = cms.bool(False),
                            NTriggeredTaus              = cms.uint32(1),
                            NTriggeredLeptons           = cms.uint32(0),
                            LeptonPDGId                 = cms.int32(0),
                            TauPDGId                    = cms.int32(15),
                            TriggerEvent                = cms.InputTag("hltTriggerSummaryAOD","",""),
                            MainFilter                  = cms.InputTag("hltFilterL3SingleTau","","HLT"),
                            L1BackupFilter              = cms.InputTag("NONE"),
                            L2BackupFilter              = cms.InputTag("NONE"),
                            L25BackupFilter             = cms.InputTag("NONE"),
                            L3BackupFilter              = cms.InputTag("NONE"),
                            refTauObjects               = cms.InputTag("TauRefProducer","PFTaus"),
                            refLeptonObjects            = cms.InputTag("NONE","NONE"),
                            matchingDeltaR              = cms.double(0.3),
                            HistEtMin                   = cms.double(0.),
                            HistEtMax                   = cms.double(100),
                            HistNEtBins                 = cms.int32(50),
                            HistNEtaBins                = cms.int32(25)
                            )

#-----------L2 Monitoring Ref PFTaus
SingleTauDQML2RefPFTaus = cms.EDFilter("HLTTauCaloDQMOfflineSource",
                                  DQMFolder              = cms.string('HLTOffline/HLTTAU/SingleTau/L2_Matched_To_PFTau'),
                                  L2InfoAssociationInput = cms.InputTag("hltL2SingleTauIsolationProducer","L2TauIsolationInfoAssociator"),
                                  refCollection          = cms.InputTag("TauRefProducer","PFTaus"),
                                  MET                    = cms.InputTag("hltMet"),
                                  doReference            = cms.bool(True),
                                  MatchDeltaR            = cms.double(0.3),
                                  OutputFileName         = cms.string(''),
                                  L2IsolatedJets         = cms.InputTag("hltL2SingleTauIsolationSelector","Isolated"),
                                  EtMin                  = cms.double(0.),
                                  EtMax                  = cms.double(100.),
                                  NBins                  = cms.int32(50)
                              )

#----------L2Monitoring Ref CaloTaus
SingleTauDQML2RefCaloTaus = cms.EDFilter("HLTTauCaloDQMOfflineSource",
                                  DQMFolder              = cms.string('HLTOffline/HLTTAU/SingleTau/L2_Matched_To_CaloTau'),
                                  L2InfoAssociationInput = cms.InputTag("hltL2SingleTauIsolationProducer","L2TauIsolationInfoAssociator"),
                                  refCollection          = cms.InputTag("TauRefProducer","CaloTaus"),
                                  MET                    = cms.InputTag("hltMet"),
                                  doReference            = cms.bool(True),
                                  MatchDeltaR            = cms.double(0.3),
                                  OutputFileName         = cms.string(''),
                                  L2IsolatedJets         = cms.InputTag("hltL2SingleTauIsolationSelector","Isolated"),
                                  EtMin                  = cms.double(0.),
                                  EtMax                  = cms.double(100.),
                                  NBins                  = cms.int32(50)
                              )
#---------L2 Monitoring Ref Electrons
SingleTauDQML2RefElectrons = cms.EDFilter("HLTTauCaloDQMOfflineSource",
                                  DQMFolder              = cms.string('HLTOffline/HLTTAU/SingleTau/L2_Matched_To_Electrons'),
                                  L2InfoAssociationInput = cms.InputTag("hltL2SingleTauIsolationProducer","L2TauIsolationInfoAssociator"),
                                  refCollection          = cms.InputTag("TauRefProducer","Electrons"),
                                  MET                    = cms.InputTag("hltMet"),
                                  doReference            = cms.bool(True),
                                  MatchDeltaR            = cms.double(0.3),
                                  OutputFileName         = cms.string(''),
                                  L2IsolatedJets         = cms.InputTag("hltL2SingleTauIsolationSelector","Isolated"),
                                  EtMin                  = cms.double(0.),
                                  EtMax                  = cms.double(100.),
                                  NBins                  = cms.int32(50)
                              )










#---------L3Monitoring Ref PFTaus
SingleTauDQML3RefPFTaus = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/SingleTau/L3_Matched_To_PFTaus'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL3SingleTau"),
                                   InputJets              = cms.InputTag("hltIsolatedL25SingleTau"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL3PixelTau"),
                                   refCollection          = cms.InputTag("TauRefProducer","PFTaus"),
                                   Type                   = cms.string('L3'),
                                   doReference            = cms.bool(True),
                                   MatchDeltaR            = cms.double(0.3),
                                   OutputFileName         = cms.string(''),
                                   EtMin                  = cms.double(0.),
                                   EtMax                  = cms.double(200.),
                                   NBins                  = cms.int32(50)
                               )

#---------L3Monitoring Ref CaloTaus
SingleTauDQML3RefCaloTaus = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/SingleTau/L3_Matched_To_Calotaus'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL3SingleTau"),
                                   InputJets              = cms.InputTag("hltIsolatedL25SingleTau"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL3SingleTau"),
                                   refCollection          = cms.InputTag("TauRefProducer","CaloTaus"),
                                   Type                   = cms.string('L3'),
                                   doReference            = cms.bool(True),
                                   MatchDeltaR            = cms.double(0.3),
                                   OutputFileName         = cms.string(''),
                                   EtMin                  = cms.double(0.),
                                   EtMax                  = cms.double(100.),
                                   NBins                  = cms.int32(50)
                               )

#---------L3Monitoring Ref Electrons
SingleTauDQML3RefElectrons = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/SingleTau/L3_Matched_To_Electrons'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL3SingleTauIsolated"),
                                   InputJets              = cms.InputTag("hltIsolatedL25SingleTau"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL3SingleTau"),
                                   refCollection          = cms.InputTag("TauRefProducer","Electrons"),
                                   Type                   = cms.string('L3'),
                                   doReference            = cms.bool(True),
                                   MatchDeltaR            = cms.double(0.3),
                                   OutputFileName         = cms.string(''),
                                   EtMin                  = cms.double(0.),
                                   EtMax                  = cms.double(100.),
                                   NBins                  = cms.int32(50)
                               )





#---------L25Monitoring Ref PFTaus
SingleTauDQML25RefPFTaus = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/SingleTau/L25_Matched_To_PFtaus'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL25SingleTau"),
                                   InputJets              = cms.InputTag("hltL2SingleTauIsolationSelector","Isolated"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL25SingleTau"),
                                   refCollection          = cms.InputTag("TauRefProducer","PFTaus"),
                                   Type                   = cms.string('L25'),
                                   doReference            = cms.bool(True),
                                   MatchDeltaR            = cms.double(0.3),
                                   OutputFileName         = cms.string(''),
                                   EtMin                  = cms.double(0.),
                                   EtMax                  = cms.double(200.),
                                   NBins                  = cms.int32(50)
                               )

#---------L25Monitoring Ref CaloTaus
SingleTauDQML25RefCaloTaus = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/SingleTau/L25_Matched_To_Calotaus'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL25SingleTau"),
                                   InputJets              = cms.InputTag("hltL2SingleTauIsolationSelector","Isolated"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL25SingleTau"),
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
SingleTauDQML25RefElectrons = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/SingleTau/L25_Matched_To_Electrons'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL25SingleTauIsolated"),
                                   InputJets              = cms.InputTag("hltL2SingleTauIsolationSelector","Isolated"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL25SingleTau"),
                                   refCollection          = cms.InputTag("TauRefProducer","Electrons"),
                                   Type                   = cms.string('L25'),
                                   doReference            = cms.bool(True),
                                   MatchDeltaR            = cms.double(0.3),
                                   OutputFileName         = cms.string(''),
                                   EtMin                  = cms.double(0.),
                                   EtMax                  = cms.double(100.),
                                   NBins                  = cms.int32(50)
                               )


HLTTauOfflineDQMSingleTauSequence = cms.Sequence(SingleTauDQMPFTaus+SingleTauDQML2RefPFTaus+SingleTauDQML2RefCaloTaus+SingleTauDQML2RefElectrons+SingleTauDQML25RefPFTaus+SingleTauDQML25RefCaloTaus+SingleTauDQML25RefElectrons+SingleTauDQML3RefPFTaus+SingleTauDQML3RefCaloTaus+SingleTauDQML3RefElectrons)








