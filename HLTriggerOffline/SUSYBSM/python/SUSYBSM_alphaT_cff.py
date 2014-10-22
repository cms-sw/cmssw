import FWCore.ParameterSet.Config as cms

SUSY_HLT_HT200_alphaT0p57 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'TEST'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  #pfMETCollection = cms.InputTag("pfMet"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  #caloJetCollection = cms.InputTag("hltAK4CaloJets"),
  #pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','TEST'),
  HLTProcess = cms.string('TEST'),
  TriggerPath = cms.string('HLT_HT200_DiJet90_AlphaT0p57_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoTkMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT200CaloAlphaT0p57', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(50.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_HT250_alphaT0p55 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'TEST'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  #pfMETCollection = cms.InputTag("pfMet"),
  #pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  #caloJetCollection = cms.InputTag("hltAK4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('TEST'),
  TriggerPath = cms.string('HLT_HT250_DiJet90_AlphaT0p55_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoTkMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT250CaloAlphaT0p55', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(50.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_HT300_alphaT0p53 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'TEST'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  #pfMETCollection = cms.InputTag("pfMet"),
  #pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  #caloJetCollection = cms.InputTag("hltAK4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('TEST'),
  TriggerPath = cms.string('HLT_HT300_DiJet90_AlphaT0p53_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoTkMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT300CaloAlphaT0p53', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(50.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_HT350_alphaT0p52 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'TEST'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  #pfMETCollection = cms.InputTag("pfMet"),
  #pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  #caloJetCollection = cms.InputTag("hltAK4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('TEST'),
  TriggerPath = cms.string('HLT_HT350_DiJet90_AlphaT0p52_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoTkMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT350CaloAlphaT0p52', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(50.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_HT400_alphaT0p51 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'TEST'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  #pfMETCollection = cms.InputTag("pfMet"),
  #pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  #caloJetCollection = cms.InputTag("hltAK4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('TEST'),
  TriggerPath = cms.string('HLT_HT400_DiJet90_AlphaT0p51_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoTkMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT400CaloAlphaT0p51', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(50.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_alphaT_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLTriggerOffline/SUSYBSM/HLT_HT200_DiJet90_AlphaT0p57_v1",
        "HLTriggerOffline/SUSYBSM/HLT_HT250_DiJet90_AlphaT0p55_v1",
        "HLTriggerOffline/SUSYBSM/HLT_HT300_DiJet90_AlphaT0p53_v1",
        "HLTriggerOffline/SUSYBSM/HLT_HT350_DiJet90_AlphaT0p52_v1",
        "HLTriggerOffline/SUSYBSM/HLT_HT400_DiJet90_AlphaT0p51_v1",
        ),
    
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "htTurnOn_eff 'Turn-on vs HT; PFHT (GeV); #epsilon' htTurnOn_num htTurnOn_den",
       "alphaTTurnOn_eff 'Turn-on vs alpha T; PFMET (GeV); #epsilon' alphaTTurnOn_num alphaTTurnOn_den",
    )
)
