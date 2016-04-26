import FWCore.ParameterSet.Config as cms

SUSY_HLT_HT_DoubleMuon = cms.EDAnalyzer("SUSY_HLT_DoubleMuon_Hadronic",
  trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),
  MuonCollection = cms.InputTag("muons"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_DoubleMu8_Mass8_PFHT300_v'),
  TriggerPathAuxiliaryForMuon = cms.string('HLT_PFHT800_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_DoubleMu33NoFiltersNoVtx_v'),
  TriggerFilter = cms.InputTag('hltDoubleMu8Mass8L3Filtered','','HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(30.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_HT250_DoubleMuon = cms.EDAnalyzer("SUSY_HLT_DoubleMuon_Hadronic",
  trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),
  MuonCollection = cms.InputTag("muons"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_DoubleMu8_Mass8_PFHT250_v'),
  TriggerPathAuxiliaryForMuon = cms.string('HLT_PFHT800_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_DoubleMu33NoFiltersNoVtx_v'),
  TriggerFilter = cms.InputTag('hltDoubleMu8Mass8L3Filtered','','HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(30.0),
  EtaThrJet = cms.untracked.double(3.0)
)


SUSY_HLT_HT_DoubleMuon_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_DoubleMu8_Mass8_PFHT300_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfHTTurnOn_eff 'Turn-on vs HT; PFHT (GeV); #epsilon' pfHTTurnOn_num pfHTTurnOn_den",
       "MuTurnOn_eff 'Turn-on vs Mu pT; pT (GeV); #epsilon' MuTurnOn_num MuTurnOn_den",
    )
)

SUSY_HLT_HT250_DoubleMuon_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_DoubleMu8_Mass8_PFHT250_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfHTTurnOn_eff 'Turn-on vs HT; PFHT (GeV); #epsilon' pfHTTurnOn_num pfHTTurnOn_den",
       "MuTurnOn_eff 'Turn-on vs Mu pT; pT (GeV); #epsilon' MuTurnOn_num MuTurnOn_den",
    )
)
