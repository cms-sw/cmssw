import FWCore.ParameterSet.Config as cms

SUSY_HLT_Electron_BJet = cms.EDAnalyzer("SUSY_HLT_Electron_BJet",
  trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),
  ElectronCollection = cms.InputTag("gedGsfElectrons"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Ele10_CaloIdL_TrackIdVL_CentralPFJet30_BTagCSV0p5PF_v'),
  TriggerFilterEle = cms.InputTag('hltSingleEle10CaloIdTrackIdVLDphiFilter','','HLT'), #the last filter in the sequence HLTSingleEle10CaloIdTrkIdVLSequence 
  TriggerFilterJet = cms.InputTag('hltCSV0p5FilterSingleEle10','','HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)


SUSY_HLT_Electron_BJet_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Electron_BJet",
  trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),
  ElectronCollection = cms.InputTag("gedGsfElectrons"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Ele10_CaloIdL_TrackIdVL_CentralPFJet30_BTagCSV0p5PF_v'),
  TriggerFilterEle = cms.InputTag('hltSingleEle10CaloIdTrackIdVLDphiFilter','','HLT'),  
  TriggerFilterJet = cms.InputTag('hltCSV0p5FilterSingleEle10','','HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)


SUSY_HLT_Electron_BJet_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Ele10_CaloIdL_TrackIdVL_CentralPFJet30_BTagCSV0p5PF"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfHTTurnOn_eff 'Turn-on vs HT; PFHT (GeV); #epsilon' pfHTTurnOn_num pfHTTurnOn_den",
       "EleTurnOn_eff 'Turn-on vs Mu pT; pT (GeV); #epsilon' EleTurnOn_num EleTurnOn_den",
    )
)


SUSY_HLT_Electron_BJet_FASTSIM_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Ele10_CaloIdL_TrackIdVL_CentralPFJet30_BTagCSV0p5PF"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfHTTurnOn_eff 'Turn-on vs HT; PFHT (GeV); #epsilon' pfHTTurnOn_num pfHTTurnOn_den",
       "EleTurnOn_eff 'Turn-on vs Mu pT; pT (GeV); #epsilon' EleTurnOn_num EleTurnOn_den",
    )
)
