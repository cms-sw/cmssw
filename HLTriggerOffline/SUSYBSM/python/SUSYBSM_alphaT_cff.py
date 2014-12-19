import FWCore.ParameterSet.Config as cms

SUSY_HLT_HT200_alphaT0p57 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_PFHT200_PFDiJet90_PFAlphaT0p57_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter = cms.InputTag('hltHT150CaloAlphaT0p54', '', 'HLT'),
  TriggerFilter = cms.InputTag('hltHT200PFAlphaT0p57', '', 'HLT'),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  pfAlphaTThrTurnon = cms.untracked.double(0.59),
  pfHtThrTurnon = cms.untracked.double(225),
  caloAlphaTThrTurnon = cms.untracked.double(0.57),
  caloHtThrTurnon = cms.untracked.double(200),
)

SUSY_HLT_HT250_alphaT0p55 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_PFHT250_PFDiJet90_PFAlphaT0p55_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter = cms.InputTag('hltHT200CaloAlphaT0p535', '', 'HLT'),
  TriggerFilter = cms.InputTag('hltHT250PFAlphaT0p55', '', 'HLT'),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  pfAlphaTThrTurnon = cms.untracked.double(0.57),
  pfHtThrTurnon = cms.untracked.double(275),
  caloAlphaTThrTurnon = cms.untracked.double(0.55),
  caloHtThrTurnon = cms.untracked.double(200),
)

SUSY_HLT_HT300_alphaT0p53 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_PFHT300_PFDiJet90_PFAlphaT0p53_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter = cms.InputTag('hltHT250CaloAlphaT0p525', '', 'HLT'), 
  TriggerFilter = cms.InputTag('hltHT300PFAlphaT0p53', '', 'HLT'),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  pfAlphaTThrTurnon = cms.untracked.double(0.55),
  pfHtThrTurnon = cms.untracked.double(325),
  caloAlphaTThrTurnon = cms.untracked.double(0.53),
  caloHtThrTurnon = cms.untracked.double(300),
)

SUSY_HLT_HT350_alphaT0p52 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),#ak4PFJetsCHS
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_PFHT350_PFDiJet90_PFAlphaT0p52_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter = cms.InputTag('hltHT300CaloAlphaT0p52', '', 'HLT'),
  TriggerFilter = cms.InputTag('hltHT350PFAlphaT0p52', '', 'HLT'),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  pfAlphaTThrTurnon = cms.untracked.double(0.54),
  pfHtThrTurnon = cms.untracked.double(375),
  caloAlphaTThrTurnon = cms.untracked.double(0.52),
  caloHtThrTurnon = cms.untracked.double(350),
)

SUSY_HLT_HT400_alphaT0p51 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_PFHT400_PFDiJet90_PFAlphaT0p51_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter = cms.InputTag('hltHT375CaloAlphaT0p51', '', 'HLT'),
  TriggerFilter = cms.InputTag('hltHT400PFAlphaT0p51', '', 'HLT'),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  pfAlphaTThrTurnon = cms.untracked.double(0.53),
  pfHtThrTurnon = cms.untracked.double(425),
  caloAlphaTThrTurnon = cms.untracked.double(0.51),
  caloHtThrTurnon = cms.untracked.double(400),
)

SUSY_HLT_alphaT_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_PFHT200_PFDiJet90_PFAlphaT0p57_v",
        "HLT/SUSYBSM/HLT_PFHT250_PFDiJet90_PFAlphaT0p55_v",
        "HLT/SUSYBSM/HLT_PFHT300_PFDiJet90_PFAlphaT0p53_v",
        "HLT/SUSYBSM/HLT_PFHT350_PFDiJet90_PFAlphaT0p52_v",
        "HLT/SUSYBSM/HLT_PFHT400_PFDiJet90_PFAlphaT0p51_v",
        ),
    
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfHtTurnOn_eff 'Turn-on vs PF HT; HT (GeV); #epsilon' pfHtTurnOn_num pfHtTurnOn_den",
       "pfAlphaTTurnOn_eff 'Turn-on vs PF alpha T; AlphaT (GeV); #epsilon' pfAlphaTTurnOn_num pfAlphaTTurnOn_den",
       "caloHtTurnOn_eff 'Turn-on vs Calo HT; HT (GeV); #epsilon' caloHtTurnOn_num caloHtTurnOn_den",
       "caloAlphaTTurnOn_eff 'Turn-on vs Calo alpha T; AlphaT (GeV); #epsilon' caloAlphaTTurnOn_num caloAlphaTTurnOn_den",
    )
)
