import FWCore.ParameterSet.Config as cms
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

                                              leptonFilter = cms.InputTag('hltEle15VVVLGsfTrackIsoFilter','','HLT'),
                                              hltHt = cms.InputTag('hltPFHTJet30','','HLT'),
                                              hltMet = cms.InputTag(''),
                                              hltJets = cms.InputTag(''),
                                              hltJetTags = cms.InputTag(''),

                                              triggerResults = cms.InputTag('TriggerResults','','HLT'),
                                              trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

                                              hltProcess = cms.string('HLT'),

                                              triggerPath = cms.string('HLT_Ele15_IsoVVVL_PFHT600_v'),
                                              triggerPathAuxiliary = cms.string('HLT_Ele38_WPTight_Gsf_v'),
                                              triggerPathLeptonAuxiliary = cms.string('HLT_PFHT1050_v'),

                                              csvlCut = cms.untracked.double(0.244),
                                              csvmCut = cms.untracked.double(0.679),
                                              csvtCut = cms.untracked.double(0.898),

                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(-1.0),
        
                                              nels = cms.untracked.double(1),
                                              nmus = cms.untracked.double(0),
                                            
                                              leptonPtThreshold = cms.untracked.double(30.0),
                                              htThreshold = cms.untracked.double(1100.),
                                              metThreshold = cms.untracked.double(-1.0),
                                              csvThreshold = cms.untracked.double(-1.0)
                                              )

DQMOffline_Ele15_HT600_POSTPROCESSING = cms.EDAnalyzer('DQMGenericClient',
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_Ele15_IsoVVVL_PFHT600_v'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Electron p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den"
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

                                              leptonFilter = cms.InputTag('hltL3MuVVVLIsoFIlter','','HLT'),
                                              hltHt = cms.InputTag('hltPFHTJet30','','HLT'),
                                              hltMet = cms.InputTag(''),
                                              hltJets = cms.InputTag(''),
                                              hltJetTags = cms.InputTag(''),

                                              triggerResults = cms.InputTag('TriggerResults','','HLT'),
                                              trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

                                              hltProcess = cms.string('HLT'),

                                              triggerPath = cms.string('HLT_Mu15_IsoVVVL_PFHT600_v'),
                                              triggerPathAuxiliary = cms.string('HLT_IsoMu27_v'),
                                              triggerPathLeptonAuxiliary = cms.string('HLT_PFHT1050_v'),

                                              csvlCut = cms.untracked.double(0.244),
                                              csvmCut = cms.untracked.double(0.679),
                                              csvtCut = cms.untracked.double(0.898),

                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(-1.0),
        
                                              nels = cms.untracked.double(0),
                                              nmus = cms.untracked.double(1),
                                            
                                              leptonPtThreshold = cms.untracked.double(30.0),
                                              htThreshold = cms.untracked.double(1100.),
                                              metThreshold = cms.untracked.double(-1.0),
                                              csvThreshold = cms.untracked.double(-1.0)
                                              )

DQMOffline_Mu15_HT600_POSTPROCESSING = cms.EDAnalyzer('DQMGenericClient',
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_Mu15_IsoVVVL_PFHT600_v'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
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

                                              leptonFilter = cms.InputTag('hltEle15VVVLGsfTrackIsoFilter','','HLT'),
                                              hltHt = cms.InputTag('hltPFHTJet30','','HLT'),
                                              hltMet = cms.InputTag(''),
                                              hltJets = cms.InputTag(''),
                                              hltJetTags = cms.InputTag(''),

                                              triggerResults = cms.InputTag('TriggerResults','','HLT'),
                                              trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

                                              hltProcess = cms.string('HLT'),

                                              triggerPath = cms.string('HLT_Ele15_IsoVVVL_PFHT450_v'),
                                              triggerPathAuxiliary = cms.string('HLT_Ele38_WPTight_Gsf_v'),
                                              triggerPathLeptonAuxiliary = cms.string('HLT_PFHT1050_v'),

                                              csvlCut = cms.untracked.double(0.244),
                                              csvmCut = cms.untracked.double(0.679),
                                              csvtCut = cms.untracked.double(0.898),

                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1),
                                              htCut = cms.untracked.double(-1),
            
                                              nels = cms.untracked.double(1),
                                              nmus = cms.untracked.double(0),
                                            
                                              leptonPtThreshold = cms.untracked.double(30.0),
                                              htThreshold = cms.untracked.double(1100.),
                                              metThreshold = cms.untracked.double(-1.0),
                                              csvThreshold = cms.untracked.double(-1.0)
                                              )

DQMOffline_Ele15_HT450_POSTPROCESSING = cms.EDAnalyzer('DQMGenericClient',
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_Ele15_IsoVVVL_PFHT450_v'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Electron p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den"
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

                                              leptonFilter = cms.InputTag('hltL3MuVVVLIsoFIlter','','HLT'),
                                              hltHt = cms.InputTag('hltPFHTJet30','','HLT'),
                                              hltMet = cms.InputTag(''),
                                              hltJets = cms.InputTag(''),
                                              hltJetTags = cms.InputTag(''),

                                              triggerResults = cms.InputTag('TriggerResults','','HLT'),
                                              trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

                                              hltProcess = cms.string('HLT'),

                                              triggerPath = cms.string('HLT_Mu15_IsoVVVL_PFHT450_v'),
                                              triggerPathAuxiliary = cms.string('HLT_IsoMu27_v'),
                                              triggerPathLeptonAuxiliary = cms.string('HLT_PFHT1050_v'),

                                              csvlCut = cms.untracked.double(0.244),
                                              csvmCut = cms.untracked.double(0.679),
                                              csvtCut = cms.untracked.double(0.898),

                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(-1.0),
        
                                              nels = cms.untracked.double(0),
                                              nmus = cms.untracked.double(1),
                                            
                                              leptonPtThreshold = cms.untracked.double(30.0),
                                              htThreshold = cms.untracked.double(1100.),
                                              metThreshold = cms.untracked.double(-1.0),
                                              csvThreshold = cms.untracked.double(-1.0)
                                              )

DQMOffline_Mu15_HT450_POSTPROCESSING = cms.EDAnalyzer('DQMGenericClient',
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_Mu15_IsoVVVL_PFHT450_v'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
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

                                              leptonFilter = cms.InputTag('hltEle50VVVLGsfTrackIsoFilter','','HLT'),
                                              hltHt = cms.InputTag('hltPFHTJet30','','HLT'),
                                              hltMet = cms.InputTag(''),
                                              hltJets = cms.InputTag(''),
                                              hltJetTags = cms.InputTag(''),

                                              triggerResults = cms.InputTag('TriggerResults','','HLT'),
                                              trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

                                              hltProcess = cms.string('HLT'),

                                              triggerPath = cms.string('HLT_Ele50_IsoVVVL_PFHT450_v'),
                                              triggerPathAuxiliary = cms.string('HLT_Ele38_WPTight_Gsf_v'),
                                              triggerPathLeptonAuxiliary = cms.string('HLT_PFHT1050_v'),

                                              csvlCut = cms.untracked.double(0.244),
                                              csvmCut = cms.untracked.double(0.679),
                                              csvtCut = cms.untracked.double(0.898),

                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(-1.),
        
                                              nels = cms.untracked.double(1),
                                              nmus = cms.untracked.double(0),
                                            
                                              leptonPtThreshold = cms.untracked.double(60.0),
                                              htThreshold = cms.untracked.double(1100.0),
                                              metThreshold = cms.untracked.double(-1.0),
                                              csvThreshold = cms.untracked.double(-1.0)
                                              )

DQMOffline_Ele50_HT450_POSTPROCESSING = cms.EDAnalyzer('DQMGenericClient',
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_Ele50_IsoVVVL_PFHT450_v'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Electron p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den"
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

                                              leptonFilter = cms.InputTag('hltL3MuVVVLIsoFIlter','','HLT'),
                                              hltHt = cms.InputTag('hltPFHTJet30','','HLT'),
                                              hltMet = cms.InputTag(''),
                                              hltJets = cms.InputTag(''),
                                              hltJetTags = cms.InputTag(''),

                                              triggerResults = cms.InputTag('TriggerResults','','HLT'),
                                              trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

                                              hltProcess = cms.string('HLT'),

                                              triggerPath = cms.string('HLT_Mu50_IsoVVVL_PFHT450_v'),
                                              triggerPathAuxiliary = cms.string('HLT_IsoMu27_v'),
                                              triggerPathLeptonAuxiliary = cms.string('HLT_PFHT1050_v'),

                                              csvlCut = cms.untracked.double(0.244),
                                              csvmCut = cms.untracked.double(0.679),
                                              csvtCut = cms.untracked.double(0.898),

                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(-1.),
        
                                              nels = cms.untracked.double(0),
                                              nmus = cms.untracked.double(1),
                                            
                                              leptonPtThreshold = cms.untracked.double(60.0),
                                              htThreshold = cms.untracked.double(1100.0),
                                              metThreshold = cms.untracked.double(-1.0),
                                              csvThreshold = cms.untracked.double(-1.0)
                                              )

DQMOffline_Mu50_HT450_POSTPROCESSING = cms.EDAnalyzer('DQMGenericClient',
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_Mu50_IsoVVVL_PFHT450_v'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
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

                                              leptonFilter = cms.InputTag('hltEle15VVVLGsfTrackIsoFilter','','HLT'),
                                              hltHt = cms.InputTag('hltPFHTJet30','','HLT'),
                                              hltMet = cms.InputTag(''),
                                              hltJets = cms.InputTag(''),
                                              hltJetTags = cms.InputTag(''),

                                              triggerResults = cms.InputTag('TriggerResults','','HLT'),
                                              trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

                                              hltProcess = cms.string('HLT'),

                                              triggerPath = cms.string('HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_v'),
                                              triggerPathAuxiliary = cms.string('HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'),
                                              triggerPathLeptonAuxiliary = cms.string('HLT_Ele15_IsoVVVL_PFHT450_v'),

                                              csvlCut = cms.untracked.double(0.244),
                                              csvmCut = cms.untracked.double(0.679),
                                              csvtCut = cms.untracked.double(0.898),

                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(-1.),
        
                                              nels = cms.untracked.double(1),
                                              nmus = cms.untracked.double(0),
                                            
                                              leptonPtThreshold = cms.untracked.double(20.0),
                                              htThreshold = cms.untracked.double(500.0),
                                              metThreshold = cms.untracked.double(-1.0),
                                              csvThreshold = cms.untracked.double(-1.0)
                                              )

DQMOffline_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_POSTPROCESSING = cms.EDAnalyzer('DQMGenericClient',
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_v'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Electron p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den"
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

                                              leptonFilter = cms.InputTag('hltL3MuVVVLIsoFIlter','','HLT'),
                                              hltHt = cms.InputTag('hltPFHTJet30','','HLT'),
                                              hltMet = cms.InputTag(''),
                                              hltJets = cms.InputTag(''),
                                              hltJetTags = cms.InputTag(''),

                                              triggerResults = cms.InputTag('TriggerResults','','HLT'),
                                              trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

                                              hltProcess = cms.string('HLT'),

                                              triggerPath = cms.string('HLT_DoubleMu4_Mass8_DZ_PFHT350_v'),
                                              triggerPathAuxiliary = cms.string('HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ'),
                                              triggerPathLeptonAuxiliary = cms.string('HLT_Mu15_IsoVVVL_PFHT450_v'),

                                              csvlCut = cms.untracked.double(0.244),
                                              csvmCut = cms.untracked.double(0.679),
                                              csvtCut = cms.untracked.double(0.898),

                                              jetPtCut = cms.untracked.double(30.0),
                                              jetEtaCut = cms.untracked.double(3.0),
                                              metCut = cms.untracked.double(-1.0),
                                              htCut = cms.untracked.double(-1.),
        
                                              nels = cms.untracked.double(0),
                                              nmus = cms.untracked.double(1),
                                            
                                              leptonPtThreshold = cms.untracked.double(20.0),
                                              htThreshold = cms.untracked.double(500.0),
                                              metThreshold = cms.untracked.double(-1.0),
                                              csvThreshold = cms.untracked.double(-1.0)
                                              )

DQMOffline_DoubleMu4_Mass8_DZ_PFHT350_POSTPROCESSING = cms.EDAnalyzer('DQMGenericClient',
                                                             subDirs = cms.untracked.vstring('HLT/LepHT/HLT_DoubleMu4_Mass8_DZ_PFHT350_v'),
                                                             efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Muon p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
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

LepHTMonitor = cms.Sequence(  DQMOffline_Ele15_HT600
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

