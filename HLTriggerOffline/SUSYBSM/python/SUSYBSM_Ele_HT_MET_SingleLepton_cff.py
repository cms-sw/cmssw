import FWCore.ParameterSet.Config as cms
from copy import deepcopy

SUSY_HLT_Ele15_HT350_MET50_SingleLepton = cms.EDAnalyzer('SUSY_HLT_SingleLepton',
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
                                                  hltMet = cms.InputTag('hltPFMETProducer','','HLT'),
                                                  hltJets = cms.InputTag(''),
                                                  hltJetTags = cms.InputTag(''),

                                                  triggerResults = cms.InputTag('TriggerResults','','HLT'),
                                                  trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

                                                  hltProcess = cms.string('HLT'),

                                                  triggerPath = cms.string('HLT_Ele15_IsoVVVL_PFHT350_PFMET50'),
                                                  triggerPathAuxiliary = cms.string('HLT_Ele35_eta2p1_WP85_Gsf_v'),
                                                  triggerPathLeptonAuxiliary = cms.string('HLT_PFHT350_PFMET120_NoiseCleaned_v'),

                                                  csvlCut = cms.untracked.double(0.244),
                                                  csvmCut = cms.untracked.double(0.679),
                                                  csvtCut = cms.untracked.double(0.898),

                                                  jetPtCut = cms.untracked.double(30.0),
                                                  jetEtaCut = cms.untracked.double(3.0),
                                                  metCut = cms.untracked.double(250.0),
                                                  htCut = cms.untracked.double(450.0),

                                                  leptonPtThreshold = cms.untracked.double(25.0),
                                                  htThreshold = cms.untracked.double(500.0),
                                                  metThreshold = cms.untracked.double(250.0),
                                                  csvThreshold = cms.untracked.double(-1.0)
                                                  )

SUSY_HLT_Ele15_HT350_MET50_SingleLepton_POSTPROCESSING = cms.EDAnalyzer('DQMGenericClient',
                                                                 subDirs = cms.untracked.vstring('HLT/SUSYBSM/HLT_Ele15_IsoVVVL_PFHT350_PFMET50'),
                                                                 efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Electron p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den",
        "pfMetTurnOn_eff ';Offline PF MET [GeV];#epsilon' pfMetTurnOn_num pfMetTurnOn_den"
        ),
                                                                 resolution = cms.vstring('')
                                                                 )

SUSY_HLT_Ele15_HT400_MET50_SingleLepton = cms.EDAnalyzer('SUSY_HLT_SingleLepton',
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
                                                  hltMet = cms.InputTag('hltPFMETProducer','','HLT'),
                                                  hltJets = cms.InputTag(''),
                                                  hltJetTags = cms.InputTag(''),

                                                  triggerResults = cms.InputTag('TriggerResults','','HLT'),
                                                  trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),

                                                  hltProcess = cms.string('HLT'),

                                                  triggerPath = cms.string('HLT_Ele15_IsoVVVL_PFHT400_PFMET50'),
                                                  triggerPathAuxiliary = cms.string('HLT_Ele35_eta2p1_WP85_Gsf_v'),
                                                  triggerPathLeptonAuxiliary = cms.string('HLT_PFHT350_PFMET120_NoiseCleaned_v'),

                                                  csvlCut = cms.untracked.double(0.244),
                                                  csvmCut = cms.untracked.double(0.679),
                                                  csvtCut = cms.untracked.double(0.898),

                                                  jetPtCut = cms.untracked.double(30.0),
                                                  jetEtaCut = cms.untracked.double(3.0),
                                                  metCut = cms.untracked.double(250.0),
                                                  htCut = cms.untracked.double(450.0),

                                                  leptonPtThreshold = cms.untracked.double(25.0),
                                                  htThreshold = cms.untracked.double(500.0),
                                                  metThreshold = cms.untracked.double(250.0),
                                                  csvThreshold = cms.untracked.double(-1.0)
                                                  )

SUSY_HLT_Ele15_HT400_MET50_SingleLepton_POSTPROCESSING = cms.EDAnalyzer('DQMGenericClient',
                                                                 subDirs = cms.untracked.vstring('HLT/SUSYBSM/HLT_Ele15_IsoVVVL_PFHT400_PFMET50'),
                                                                 efficiency = cms.vstring(
        "leptonTurnOn_eff ';Offline Electron p_{T} [GeV];#epsilon' leptonTurnOn_num leptonTurnOn_den",
        "pfHTTurnOn_eff ';Offline PF H_{T} [GeV];#epsilon' pfHTTurnOn_num pfHTTurnOn_den",
        "pfMetTurnOn_eff ';Offline PF MET [GeV];#epsilon' pfMetTurnOn_num pfMetTurnOn_den"
        ),
                                                                 resolution = cms.vstring('')
                                                                 )


# fastsim has no conversion collection (yet)
from Configuration.StandardSequences.Eras import eras
eras.fastSim.toModify(SUSY_HLT_Ele15_HT350_MET50_SingleLepton,conversionCollection=cms.InputTag(''))
eras.fastSim.toModify(SUSY_HLT_Ele15_HT400_MET50_SingleLepton,conversionCollection=cms.InputTag(''))

SUSY_HLT_Ele_HT_MET_SingleLepton = cms.Sequence( SUSY_HLT_Ele15_HT350_MET50_SingleLepton
                                                 + SUSY_HLT_Ele15_HT400_MET50_SingleLepton
)

SUSY_HLT_Ele_HT_MET_SingleLepton_POSTPROCESSING = cms.Sequence( SUSY_HLT_Ele15_HT350_MET50_SingleLepton_POSTPROCESSING
                                                                + SUSY_HLT_Ele15_HT400_MET50_SingleLepton_POSTPROCESSING
)
