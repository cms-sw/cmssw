import FWCore.ParameterSet.Config as cms

#Double Tau Path 
DoubleTauDQMPFTaus = cms.EDFilter("HLTTauDQMOfflineSource",
                            DQMFolder                   = cms.string('HLTOffline/HLTTAU/DoubleTau/'),
                            monitorName                 = cms.string('RefToPFTaus'),
                            outputFile                  = cms.string(''),
                            prescaleEvt                 = cms.int32(-1),
                            disableROOToutput           = cms.bool(True),
                            verbose                     = cms.bool(False),
                            NTriggeredTaus              = cms.uint32(2),
                            NTriggeredLeptons           = cms.uint32(0),
                            LeptonPDGId                 = cms.int32(0),
                            TauPDGId                    = cms.int32(15),
                            TriggerEvent                = cms.InputTag("hltTriggerSummaryAOD","",""),
                            MainFilter                  = cms.InputTag("hltFilterL25PixelTau","","HLT"),
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
DoubleTauDQML2RefPFTaus = cms.EDFilter("HLTTauCaloDQMOfflineSource",
                                  DQMFolder              = cms.string('HLTOffline/HLTTAU/DoubleTau/L2_Matched_To_PFTau'),
                                  L2InfoAssociationInput = cms.InputTag("hltL2DoubleTauIsolationProducer","L2TauIsolationInfoAssociator"),
                                  refCollection          = cms.InputTag("TauRefProducer","PFTaus"),
                                  MET                    = cms.InputTag("hltMet"),
                                  doReference            = cms.bool(True),
                                  MatchDeltaR            = cms.double(0.3),
                                  OutputFileName         = cms.string(''),
                                  L2IsolatedJets         = cms.InputTag("hltL2DoubleTauIsolationSelector","Isolated"),
                                  EtMin                  = cms.double(0.),
                                  EtMax                  = cms.double(100.),
                                  NBins                  = cms.int32(50)
                              )

#----------L2Monitoring Ref CaloTaus
DoubleTauDQML2RefCaloTaus = cms.EDFilter("HLTTauCaloDQMOfflineSource",
                                  DQMFolder              = cms.string('HLTOffline/HLTTAU/DoubleTau/L2_Matched_To_CaloTau'),
                                  L2InfoAssociationInput = cms.InputTag("hltL2DoubleTauIsolationProducer","L2TauIsolationInfoAssociator"),
                                  refCollection          = cms.InputTag("TauRefProducer","CaloTaus"),
                                  MET                    = cms.InputTag("hltMet"),
                                  doReference            = cms.bool(True),
                                  MatchDeltaR            = cms.double(0.3),
                                  OutputFileName         = cms.string(''),
                                  L2IsolatedJets         = cms.InputTag("hltL2DoubleTauIsolationSelector","Isolated"),
                                  EtMin                  = cms.double(0.),
                                  EtMax                  = cms.double(100.),
                                  NBins                  = cms.int32(50)
                              )
#---------L2 Monitoring Ref Electrons
DoubleTauDQML2RefElectrons = cms.EDFilter("HLTTauCaloDQMOfflineSource",
                                  DQMFolder              = cms.string('HLTOffline/HLTTAU/DoubleTau/L2_Matched_To_Electrons'),
                                  L2InfoAssociationInput = cms.InputTag("hltL2DoubleTauIsolationProducer","L2TauIsolationInfoAssociator"),
                                  refCollection          = cms.InputTag("TauRefProducer","Electrons"),
                                  MET                    = cms.InputTag("hltMet"),
                                  doReference            = cms.bool(True),
                                  MatchDeltaR            = cms.double(0.3),
                                  OutputFileName         = cms.string(''),
                                  L2IsolatedJets         = cms.InputTag("hltL2DoubleTauIsolationSelector","Isolated"),
                                  EtMin                  = cms.double(0.),
                                  EtMax                  = cms.double(100.),
                                  NBins                  = cms.int32(50)
                              )





#---------L25Monitoring Ref PFTaus
DoubleTauDQML25RefPFTaus = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/DoubleTau/L25_Matched_To_PFtaus'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL25PixelTauIsolated"),
                                   InputJets              = cms.InputTag("hltL2DoubleTauIsolationSelector","Isolated"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL25PixelTau"),
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
DoubleTauDQML25RefCaloTaus = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/DoubleTau/L25_Matched_To_Calotaus'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL25PixelTauIsolated"),
                                   InputJets              = cms.InputTag("hltL2DoubleTauIsolationSelector","Isolated"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL25PixelTau"),
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
DoubleTauDQML25RefElectrons = cms.EDFilter("HLTTauTrkDQMOfflineSource",
                                   DQMFolder              = cms.string('HLTOffline/HLTTAU/DoubleTau/L25_Matched_To_Electrons'),
                                   ConeIsolation          = cms.InputTag("hltConeIsolationL25PixelTauIsolated"),
                                   InputJets              = cms.InputTag("hltL2DoubleTauIsolationSelector","Isolated"),
                                   IsolatedJets           = cms.InputTag("hltIsolatedL25PixelTau"),
                                   refCollection          = cms.InputTag("TauRefProducer","Electrons"),
                                   Type                   = cms.string('L25'),
                                   doReference            = cms.bool(True),
                                   MatchDeltaR            = cms.double(0.3),
                                   OutputFileName         = cms.string('out.root'),
                                   EtMin                  = cms.double(0.),
                                   EtMax                  = cms.double(100.),
                                   NBins                  = cms.int32(50)
                               )

HLTTauOfflineDQMDoubleTauSequence = cms.Sequence(DoubleTauDQMPFTaus+DoubleTauDQML2RefPFTaus+DoubleTauDQML2RefCaloTaus+DoubleTauDQML2RefElectrons+DoubleTauDQML25RefPFTaus+DoubleTauDQML25RefCaloTaus+DoubleTauDQML25RefElectrons)







