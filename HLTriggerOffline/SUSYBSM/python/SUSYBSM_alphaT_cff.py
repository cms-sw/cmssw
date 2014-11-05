import FWCore.ParameterSet.Config as cms

SUSY_HLT_HT200_alphaT0p57 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  #pfMETCollection = cms.InputTag("pfMet"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  #caloJetCollection = cms.InputTag("hltAK4CaloJets"),
  #pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_HT200_DiJet90_AlphaT0p57_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoTkMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT200CaloAlphaT0p57', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  alphaTThrTurnon = cms.untracked.double(0.59),
  htThrTurnon = cms.untracked.double(225),
)

SUSY_HLT_HT250_alphaT0p55 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  #pfMETCollection = cms.InputTag("pfMet"),
  #pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  #caloJetCollection = cms.InputTag("hltAK4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_HT250_DiJet90_AlphaT0p55_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoTkMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT250CaloAlphaT0p55', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  alphaTThrTurnon = cms.untracked.double(0.57),
  htThrTurnon = cms.untracked.double(275),
)

SUSY_HLT_HT300_alphaT0p53 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  #pfMETCollection = cms.InputTag("pfMet"),
  #pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  #caloJetCollection = cms.InputTag("hltAK4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_HT300_DiJet90_AlphaT0p53_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoTkMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT300CaloAlphaT0p53', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  alphaTThrTurnon = cms.untracked.double(0.55),
  htThrTurnon = cms.untracked.double(325),
)

SUSY_HLT_HT350_alphaT0p52 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  #pfMETCollection = cms.InputTag("pfMet"),
  #pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  #caloJetCollection = cms.InputTag("hltAK4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_HT350_DiJet90_AlphaT0p52_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoTkMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT350CaloAlphaT0p52', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  alphaTThrTurnon = cms.untracked.double(0.54),
  htThrTurnon = cms.untracked.double(375),
)

SUSY_HLT_HT400_alphaT0p51 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  #pfMETCollection = cms.InputTag("pfMet"),
  #pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  #caloJetCollection = cms.InputTag("hltAK4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_HT400_DiJet90_AlphaT0p51_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoTkMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT400CaloAlphaT0p51', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  alphaTThrTurnon = cms.untracked.double(0.53),
  htThrTurnon = cms.untracked.double(425),
)

SUSY_HLT_alphaT_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_HT200_DiJet90_AlphaT0p57_v1",
        "HLT/SUSYBSM/HLT_HT250_DiJet90_AlphaT0p55_v1",
        "HLT/SUSYBSM/HLT_HT300_DiJet90_AlphaT0p53_v1",
        "HLT/SUSYBSM/HLT_HT350_DiJet90_AlphaT0p52_v1",
        "HLT/SUSYBSM/HLT_HT400_DiJet90_AlphaT0p51_v1",
        ),
    
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "htTurnOn_eff 'Turn-on vs HT; HT (GeV); #epsilon' htTurnOn_num htTurnOn_den",
       "alphaTTurnOn_eff 'Turn-on vs alpha T; AlphaT (GeV); #epsilon' alphaTTurnOn_num alphaTTurnOn_den",
    )
)
