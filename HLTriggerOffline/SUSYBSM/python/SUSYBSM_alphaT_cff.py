import FWCore.ParameterSet.Config as cms

SUSY_HLT200_alphaT0p57 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  caloMETCollection = cms.InputTag("met"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_HT200_DiJet100_AlphaT0p57_v2'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT200CaloAlphaT0p57', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(50.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT250_alphaT0p55 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  caloMETCollection = cms.InputTag("met"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_HT250_DiJet100_AlphaT0p55_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT250CaloAlphaT0p55', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(50.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT300_alphaT0p54 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  caloMETCollection = cms.InputTag("met"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_HT300_DiJet100_AlphaT0p54_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT300CaloAlphaT0p54', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(50.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_alphaT_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLTriggerOffline/SUSYBSM/HLT_HT200_DiJet100_AlphaT0p57_v2",
        "HLTriggerOffline/SUSYBSM/HLT_HT250_DiJet100_AlphaT0p55_v1",
        "HLTriggerOffline/SUSYBSM/HLT_HT300_DiJet100_AlphaT0p54_v1",
        ),
    
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "htTurnOn_eff 'Turn-on vs HT; PFHT (GeV); #epsilon' htTurnOn_num htTurnOn_den",
       "alphaTTurnOn_eff 'Turn-on vs alpha T; PFMET (GeV); #epsilon' alphaTTurnOn_num alphaTTurnOn_den",
    )
)
