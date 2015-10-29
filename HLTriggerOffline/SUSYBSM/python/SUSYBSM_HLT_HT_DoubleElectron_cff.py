import FWCore.ParameterSet.Config as cms

SUSY_HLT_HT_DoubleEle = cms.EDAnalyzer("SUSY_HLT_DoubleEle_Hadronic",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  ElectronCollection = cms.InputTag("gedGsfElectrons"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT300_v'),
  TriggerPathAuxiliaryForElectron = cms.string('HLT_PFHT800_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_MW_v'),
  TriggerFilter = cms.InputTag('hltDoubleEle8Mass8Filter', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_HT250_DoubleEle = cms.EDAnalyzer("SUSY_HLT_DoubleEle_Hadronic",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  ElectronCollection = cms.InputTag("gedGsfElectrons"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT250_v'),
  TriggerPathAuxiliaryForElectron = cms.string('HLT_PFHT800_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_MW_v'),
  TriggerFilter = cms.InputTag('hltDoubleEle8Mass8Filter', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)



SUSY_HLT_HT_DoubleEle_FASTSIM = cms.EDAnalyzer("SUSY_HLT_DoubleEle_Hadronic",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  ElectronCollection = cms.InputTag("gedGsfElectrons"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT300_v'),
  TriggerPathAuxiliaryForElectron = cms.string('HLT_PFHT800_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_MW_v'),
  TriggerFilter = cms.InputTag('hltDoubleEle8Mass8Filter', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_HT250_DoubleEle_FASTSIM = cms.EDAnalyzer("SUSY_HLT_DoubleEle_Hadronic",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  ElectronCollection = cms.InputTag("gedGsfElectrons"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT250_v'),
  TriggerPathAuxiliaryForElectron = cms.string('HLT_PFHT800_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_MW_v'),
  TriggerFilter = cms.InputTag('hltDoubleEle8Mass8Filter', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)


SUSY_HLT_HT_DoubleEle_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT300"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfHTTurnOn_eff 'Turn-on vs HT; PFHT (GeV); #epsilon' pfHTTurnOn_num pfHTTurnOn_den",
       "EleTurnOn_eff 'Turn-on vs Mu pT; pT (GeV); #epsilon' EleTurnOn_num EleTurnOn_den",
    )
)

SUSY_HLT_HT250_DoubleEle_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT250"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfHTTurnOn_eff 'Turn-on vs HT; PFHT (GeV); #epsilon' pfHTTurnOn_num pfHTTurnOn_den",
       "EleTurnOn_eff 'Turn-on vs Mu pT; pT (GeV); #epsilon' EleTurnOn_num EleTurnOn_den",
    )
)


SUSY_HLT_HT_DoubleEle_FASTSIM_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT300"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfHTTurnOn_eff 'Turn-on vs HT; PFHT (GeV); #epsilon' pfHTTurnOn_num pfHTTurnOn_den",
       "EleTurnOn_eff 'Turn-on vs Mu pT; pT (GeV); #epsilon' EleTurnOn_num EleTurnOn_den",
    )
)

SUSY_HLT_HT250_DoubleEle_FASTSIM_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT2500"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfHTTurnOn_eff 'Turn-on vs HT; PFHT (GeV); #epsilon' pfHTTurnOn_num pfHTTurnOn_den",
       "EleTurnOn_eff 'Turn-on vs Mu pT; pT (GeV); #epsilon' EleTurnOn_num EleTurnOn_den",
    )
)




