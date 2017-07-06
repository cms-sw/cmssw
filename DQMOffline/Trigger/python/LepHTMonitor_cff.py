import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from copy import deepcopy

DQMOffline_Ele15_HT600 = cms.EDAnalyzer('LepHTMonitor',
                                              electronCollection = cms.InputTag('gedGsfElectrons'),
                                              muonCollection = cms.InputTag(''),
                                              pfMetCollection = cms.InputTag('pfMet'),
                                              pfJetCollection = cms.InputTag('ak4PFJets'),
                                              jetTagCollection = cms.InputTag(''),

                                              vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                              conversionCollection = cms.InputTag('conversions'),
                                              beamSpot = cms.InputTag('offlineBeamSpot'),

                                              triggerPath = cms.string('HLT_Ele15_IsoVVVL_PFHT600'),

                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(1100.0),
                                              nels = cms.untracked.double(1),
                                              nmus = cms.untracked.double(0),
                                              leptonPtThreshold = cms.untracked.double(30.0),
                      
                                              numGenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_Ele15_IsoVVVL_PFHT600_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_lep_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_PFHT1050_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_HT_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_Ele38_WPTight_Gsf_v*","HLT_Ele27_WPTight_Gsf_v*","HLT_Ele35_WPTight_Gsf_v*","HLT_Ele40_WPTight_Gsf_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                            )

DQMOffline_Ele15_HT600_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_Ele15_IsoVVVL_PFHT600'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Electron p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "lepEtaOn_eff ';Offline Electron #eta;#epsilon' lepEtaTurnOn_num lepEtaTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den",
        ),
                                                             resolution = cms.vstring('')
                                                             )



DQMOffline_Mu15_HT600 = cms.EDAnalyzer('LepHTMonitor',
                                              electronCollection = cms.InputTag(''),
                                              muonCollection = cms.InputTag('muons'),
                                              pfMetCollection = cms.InputTag('pfMet'),
                                              pfJetCollection = cms.InputTag('ak4PFJets'),
                                              jetTagCollection = cms.InputTag(''),

                                              vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                              conversionCollection = cms.InputTag(''),
                                              beamSpot = cms.InputTag('offlineBeamSpot'),                                        

                                              triggerPath = cms.string('HLT_Mu15_IsoVVVL_PFHT600'),
        
                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(1100.0),
                                              nels = cms.untracked.double(0),
                                              nmus = cms.untracked.double(1),
                                              leptonPtThreshold = cms.untracked.double(30.0),

                                              numGenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_Mu15_IsoVVVL_PFHT600_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_lep_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_PFHT1050_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_HT_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_IsoMu27_v*","HLT_IsoMu24_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                            
                                              )

DQMOffline_Mu15_HT600_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_Mu15_IsoVVVL_PFHT600'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "lepEtaTurnOn_eff ';Offline #eta;#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den"
        ),
                                                             resolution = cms.vstring('')
                                                             )

DQMOffline_Ele15_HT450 = cms.EDAnalyzer('LepHTMonitor',
                                              electronCollection = cms.InputTag('gedGsfElectrons'),
                                              muonCollection = cms.InputTag(''),
                                              pfMetCollection = cms.InputTag('pfMet'),
                                              pfJetCollection = cms.InputTag('ak4PFJets'),
                                              jetTagCollection = cms.InputTag(''),

                                              vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                              conversionCollection = cms.InputTag('conversions'),
                                              beamSpot = cms.InputTag('offlineBeamSpot'),

                                              triggerPath = cms.string('HLT_Ele15_IsoVVVL_PFHT450'),

                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(1100.0),
                                              nels = cms.untracked.double(1),
                                              nmus = cms.untracked.double(0),
                                              leptonPtThreshold = cms.untracked.double(30.0),
                      
                                              numGenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_Ele15_IsoVVVL_PFHT450_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_lep_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_PFHT1050_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_HT_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_Ele38_WPTight_Gsf_v*","HLT_Ele27_WPTight_Gsf_v*","HLT_Ele35_WPTight_Gsf_v*","HLT_Ele40_WPTight_Gsf_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                            )

DQMOffline_Ele15_HT450_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_Ele15_IsoVVVL_PFHT450'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Electron p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "lepEtaOn_eff ';Offline Electron #eta;#epsilon' lepEtaTurnOn_num lepEtaTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den",
        ),
                                                             resolution = cms.vstring('')
                                                             )



DQMOffline_Mu15_HT450 = cms.EDAnalyzer('LepHTMonitor',
                                              electronCollection = cms.InputTag(''),
                                              muonCollection = cms.InputTag('muons'),
                                              pfMetCollection = cms.InputTag('pfMet'),
                                              pfJetCollection = cms.InputTag('ak4PFJets'),
                                              jetTagCollection = cms.InputTag(''),

                                              vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                              conversionCollection = cms.InputTag(''),
                                              beamSpot = cms.InputTag('offlineBeamSpot'),                                        

                                              triggerPath = cms.string('HLT_Mu15_IsoVVVL_PFHT450'),
        
                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(1100.0),
                                              nels = cms.untracked.double(0),
                                              nmus = cms.untracked.double(1),
                                              leptonPtThreshold = cms.untracked.double(30.0),

                                              numGenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_Mu15_IsoVVVL_PFHT450_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_lep_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_PFHT1050_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_HT_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_IsoMu27_v*","HLT_IsoMu24_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                            
                                              )

DQMOffline_Mu15_HT450_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_Mu15_IsoVVVL_PFHT450'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "lepEtaTurnOn_eff ';Offline #eta;#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den"
        ),
                                                             resolution = cms.vstring('')
                                                             )

DQMOffline_Ele50_HT450 = cms.EDAnalyzer('LepHTMonitor',
                                              electronCollection = cms.InputTag('gedGsfElectrons'),
                                              muonCollection = cms.InputTag(''),
                                              pfMetCollection = cms.InputTag('pfMet'),
                                              pfJetCollection = cms.InputTag('ak4PFJets'),
                                              jetTagCollection = cms.InputTag(''),

                                              vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                              conversionCollection = cms.InputTag('conversions'),
                                              beamSpot = cms.InputTag('offlineBeamSpot'),

                                              triggerPath = cms.string('HLT_Ele50_IsoVVVL_PFHT450'),

                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(1100.0),
                                              nels = cms.untracked.double(1),
                                              nmus = cms.untracked.double(0),
                                              leptonPtThreshold = cms.untracked.double(60.0),
                      
                                              numGenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_Ele50_IsoVVVL_PFHT450_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_lep_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_PFHT1050_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_HT_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_Ele38_WPTight_Gsf_v*","HLT_Ele27_WPTight_Gsf_v*","HLT_Ele35_WPTight_Gsf_v*","HLT_Ele40_WPTight_Gsf_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                            )

DQMOffline_Ele50_HT450_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_Ele50_IsoVVVL_PFHT450'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Electron p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "lepEtaOn_eff ';Offline Electron #eta;#epsilon' lepEtaTurnOn_num lepEtaTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den",
        ),
                                                             resolution = cms.vstring('')
                                                             )



DQMOffline_Mu50_HT450 = cms.EDAnalyzer('LepHTMonitor',
                                              electronCollection = cms.InputTag(''),
                                              muonCollection = cms.InputTag('muons'),
                                              pfMetCollection = cms.InputTag('pfMet'),
                                              pfJetCollection = cms.InputTag('ak4PFJets'),
                                              jetTagCollection = cms.InputTag(''),

                                              vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                              conversionCollection = cms.InputTag(''),
                                              beamSpot = cms.InputTag('offlineBeamSpot'),                                        

                                              triggerPath = cms.string('HLT_Mu50_IsoVVVL_PFHT450'),
        
                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(1100.0),
                                              nels = cms.untracked.double(0),
                                              nmus = cms.untracked.double(1),
                                              leptonPtThreshold = cms.untracked.double(60.0),

                                              numGenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_Mu50_IsoVVVL_PFHT450_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_lep_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_PFHT1050_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_HT_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_IsoMu27_v*","HLT_IsoMu24_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                            
                                              )

DQMOffline_Mu50_HT450_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_Mu50_IsoVVVL_PFHT450'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "lepEtaTurnOn_eff ';Offline #eta;#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den"
        ),
                                                             resolution = cms.vstring('')
                                                             )



DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350 = cms.EDAnalyzer('LepHTMonitor',
                                              electronCollection = cms.InputTag('gedGsfElectrons'),
                                              muonCollection = cms.InputTag(''),
                                              pfMetCollection = cms.InputTag('pfMet'),
                                              pfJetCollection = cms.InputTag('ak4PFJets'),
                                              jetTagCollection = cms.InputTag(''),

                                              vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                              conversionCollection = cms.InputTag('conversions'),
                                              beamSpot = cms.InputTag('offlineBeamSpot'),

                                              triggerPath = cms.string('HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350'),

                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(500.0),
                                              nels = cms.untracked.double(2),
                                              nmus = cms.untracked.double(0),
                                              leptonPtThreshold = cms.untracked.double(20.0),
                      
                                              numGenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_lep_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_Ele15_IsoVVVL_PFHT450_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_HT_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                            )

DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Electron p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "lepEtaOn_eff ';Offline Electron #eta;#epsilon' lepEtaTurnOn_num lepEtaTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den",
        ),
                                                             resolution = cms.vstring('')
                                                             )



DQMOffline_DoubleMu4_Mass8_DZ_PFHT350 = cms.EDAnalyzer('LepHTMonitor',
                                              electronCollection = cms.InputTag(''),
                                              muonCollection = cms.InputTag('muons'),
                                              pfMetCollection = cms.InputTag('pfMet'),
                                              pfJetCollection = cms.InputTag('ak4PFJets'),
                                              jetTagCollection = cms.InputTag(''),

                                              vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                              conversionCollection = cms.InputTag(''),
                                              beamSpot = cms.InputTag('offlineBeamSpot'),                                        

                                              triggerPath = cms.string('HLT_DoubleMu4_Mass8_DZ_PFHT350'),
        
                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(500.0),
                                              nels = cms.untracked.double(0),
                                              nmus = cms.untracked.double(2),
                                              leptonPtThreshold = cms.untracked.double(20.0),

                                              numGenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_DoubleMu4_Mass8_DZ_PFHT350_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_lep_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_Mu15_IsoVVVL_PFHT450_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                                              den_HT_GenericTriggerEventPSet = cms.PSet(
                                                andOr         = cms.bool( False ),
                                                andOrHlt      = cms.bool(True),# True:=OR; False:=AND
                                                hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
                                                hltPaths      = cms.vstring("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*"),
                                                errorReplyHlt = cms.bool( False ),
                                                dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
                                                dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ), # 24-27: strip, 28-29: pixel, we should add all other detectors !
                                                andOrDcs      = cms.bool( False ),
                                                errorReplyDcs = cms.bool( True ),
                                                verbosityLevel = cms.uint32(1)
                                                ),
                            
                                              )

DQMOffline_DoubleMu4_Mass8_DZ_PFHT350_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_DoubleMu4_Mass8_DZ_PFHT350'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "lepEtaTurnOn_eff ';Offline #eta;#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den"
        ),
                                                             resolution = cms.vstring('')
                                                             )



# fastsim has no conversion collection
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(DQMOffline_Ele15_HT600,conversionCollection=cms.InputTag(''))
fastSim.toModify(DQMOffline_Ele15_HT450,conversionCollection=cms.InputTag(''))
fastSim.toModify(DQMOffline_Ele50_HT450,conversionCollection=cms.InputTag(''))
fastSim.toModify(DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350,conversionCollection=cms.InputTag(''))


LepHTMonitor = cms.Sequence( DQMOffline_Ele15_HT600
                            + DQMOffline_Mu15_HT600
                            + DQMOffline_Ele15_HT450
                            + DQMOffline_Ele50_HT450
                            + DQMOffline_Mu15_HT450
                            + DQMOffline_Mu50_HT450
                            + DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350
                            + DQMOffline_DoubleMu4_Mass8_DZ_PFHT350
                            
)

LepHTClient = cms.Sequence( DQMOffline_Ele15_HT600_POSTPROCESSING
                            + DQMOffline_Mu15_HT600_POSTPROCESSING
                            + DQMOffline_Ele15_HT450_POSTPROCESSING
                            + DQMOffline_Ele50_HT450_POSTPROCESSING
                            + DQMOffline_Mu15_HT450_POSTPROCESSING
                            + DQMOffline_Mu50_HT450_POSTPROCESSING
                            + DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_POSTPROCESSING
                            + DQMOffline_DoubleMu4_Mass8_DZ_PFHT350_POSTPROCESSING
                           
)

