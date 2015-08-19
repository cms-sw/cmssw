import FWCore.ParameterSet.Config as cms


# Control trigger
SUSY_HLT_HT200_alphaT0p51 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary       = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  TriggerResults    = cms.InputTag('TriggerResults','','HLT'),        #to use with test sample
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  pfJetCollection   = cms.InputTag("ak4PFJetsCHS"),
  HLTProcess                      = cms.string('HLT'),
  TriggerPath                     = cms.string('HLT_PFHT200_PFAlphaT0p51_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter                = cms.InputTag('hltHT150CaloAlphaT0p51', '', 'HLT'),
  TriggerFilter                   = cms.InputTag('hltPFHT200PFAlphaT0p51', '', 'HLT'),
  PtThrJet            = cms.untracked.double(40.0),
  EtaThrJet           = cms.untracked.double(3.0),
  caloHtThrTurnon     = cms.untracked.double(200),
  caloAlphaTThrTurnon = cms.untracked.double(0.53),
  pfHtThrTurnon       = cms.untracked.double(225),
  pfAlphaTThrTurnon   = cms.untracked.double(0.53),
)

# Primary triggers
SUSY_HLT_HT200_alphaT0p57 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary       = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  TriggerResults    = cms.InputTag('TriggerResults','','HLT'),        #to use with test sample
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  pfJetCollection   = cms.InputTag("ak4PFJetsCHS"),
  HLTProcess                      = cms.string('HLT'),
  TriggerPath                     = cms.string('HLT_PFHT200_DiPFJetAve90_PFAlphaT0p57_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter                = cms.InputTag('hltHT150CaloAlphaT0p54', '', 'HLT'),
  TriggerFilter                   = cms.InputTag('hltPFHT200PFAlphaT0p57', '', 'HLT'),
  PtThrJet            = cms.untracked.double(40.0),
  EtaThrJet           = cms.untracked.double(3.0),
  caloHtThrTurnon     = cms.untracked.double(200),
  caloAlphaTThrTurnon = cms.untracked.double(0.61),
  pfHtThrTurnon       = cms.untracked.double(225),
  pfAlphaTThrTurnon   = cms.untracked.double(0.65),
)

SUSY_HLT_HT250_alphaT0p55 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_PFHT250_DiPFJetAve90_PFAlphaT0p55_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter = cms.InputTag('hltHT225CaloAlphaT0p53', '', 'HLT'),
  TriggerFilter    = cms.InputTag('hltPFHT250PFAlphaT0p55', '', 'HLT'),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  pfAlphaTThrTurnon = cms.untracked.double(0.6),
  pfHtThrTurnon = cms.untracked.double(275),
  caloAlphaTThrTurnon = cms.untracked.double(0.57),
  caloHtThrTurnon = cms.untracked.double(250),
)

SUSY_HLT_HT300_alphaT0p53 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_PFHT300_DiPFJetAve90_PFAlphaT0p53_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter = cms.InputTag('hltHT275CaloAlphaT0p525', '', 'HLT'), 
  TriggerFilter    = cms.InputTag('hltPFHT300PFAlphaT0p53', '', 'HLT'),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  pfAlphaTThrTurnon = cms.untracked.double(0.56),
  pfHtThrTurnon = cms.untracked.double(325),
  caloAlphaTThrTurnon = cms.untracked.double(0.55),
  caloHtThrTurnon = cms.untracked.double(300),
)

SUSY_HLT_HT350_alphaT0p52 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),#ak4PFJetsCHS
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_PFHT350_DiPFJetAve90_PFAlphaT0p52_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter = cms.InputTag('hltHT325CaloAlphaT0p515', '', 'HLT'),
  TriggerFilter    = cms.InputTag('hltPFHT350PFAlphaT0p52', '', 'HLT'),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  pfAlphaTThrTurnon = cms.untracked.double(0.55),
  pfHtThrTurnon = cms.untracked.double(375),
  caloAlphaTThrTurnon = cms.untracked.double(0.53),
  caloHtThrTurnon = cms.untracked.double(350),
)

SUSY_HLT_HT400_alphaT0p51 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_PFHT400_DiPFJetAve90_PFAlphaT0p51_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter = cms.InputTag('hltHT375CaloAlphaT0p51', '', 'HLT'),
  TriggerFilter    = cms.InputTag('hltPFHT400PFAlphaT0p51', '', 'HLT'),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  pfAlphaTThrTurnon = cms.untracked.double(0.54),
  pfHtThrTurnon = cms.untracked.double(425),
  caloAlphaTThrTurnon = cms.untracked.double(0.53),
  caloHtThrTurnon = cms.untracked.double(400),
)

# Backup triggers
SUSY_HLT_HT200_alphaT0p63 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary       = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  TriggerResults    = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  pfJetCollection   = cms.InputTag("ak4PFJetsCHS"),
  HLTProcess                      = cms.string('HLT'),
  TriggerPath                     = cms.string('HLT_PFHT200_DiPFJetAve90_PFAlphaT0p63_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter                = cms.InputTag('hltHT175CaloAlphaT0p59', '', 'HLT'),
  TriggerFilter                   = cms.InputTag('hltPFHT200PFAlphaT0p63', '', 'HLT'),
  PtThrJet            = cms.untracked.double(40.0),
  EtaThrJet           = cms.untracked.double(3.0),
  caloHtThrTurnon     = cms.untracked.double(200),
  caloAlphaTThrTurnon = cms.untracked.double(0.61),
  pfHtThrTurnon       = cms.untracked.double(225),
  pfAlphaTThrTurnon   = cms.untracked.double(0.65),
)

SUSY_HLT_HT250_alphaT0p58 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_PFHT250_DiPFJetAve90_PFAlphaT0p58_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter = cms.InputTag('hltHT225CaloAlphaT0p55', '', 'HLT'),
  TriggerFilter = cms.InputTag('hltPFHT250PFAlphaT0p58', '', 'HLT'),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  pfAlphaTThrTurnon = cms.untracked.double(0.6),
  pfHtThrTurnon = cms.untracked.double(275),
  caloAlphaTThrTurnon = cms.untracked.double(0.57),
  caloHtThrTurnon = cms.untracked.double(250),
)

SUSY_HLT_HT300_alphaT0p54 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_PFHT300_DiPFJetAve90_PFAlphaT0p54_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter = cms.InputTag('hltHT250CaloAlphaT0p53', '', 'HLT'), 
  TriggerFilter = cms.InputTag('hltPFHT300PFAlphaT0p54', '', 'HLT'),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  pfAlphaTThrTurnon = cms.untracked.double(0.56),
  pfHtThrTurnon = cms.untracked.double(325),
  caloAlphaTThrTurnon = cms.untracked.double(0.55),
  caloHtThrTurnon = cms.untracked.double(300),
)

SUSY_HLT_HT350_alphaT0p53 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),#ak4PFJetsCHS
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_PFHT350_DiPFJetAve90_PFAlphaT0p53_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter = cms.InputTag('hltHT300CaloAlphaT0p51', '', 'HLT'),
  TriggerFilter = cms.InputTag('hltPFHT350PFAlphaT0p53', '', 'HLT'),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  pfAlphaTThrTurnon = cms.untracked.double(0.55),
  pfHtThrTurnon = cms.untracked.double(375),
  caloAlphaTThrTurnon = cms.untracked.double(0.53),
  caloHtThrTurnon = cms.untracked.double(350),
)

SUSY_HLT_HT400_alphaT0p52 = cms.EDAnalyzer("SUSY_HLT_alphaT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_PFHT400_DiPFJetAve90_PFAlphaT0p52_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerPreFilter = cms.InputTag('hltHT325CaloAlphaT0p51', '', 'HLT'),
  TriggerFilter = cms.InputTag('hltPFHT400PFAlphaT0p52', '', 'HLT'),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0),
  pfAlphaTThrTurnon = cms.untracked.double(0.54),
  pfHtThrTurnon = cms.untracked.double(425),
  caloAlphaTThrTurnon = cms.untracked.double(0.53),
  caloHtThrTurnon = cms.untracked.double(400),
)

SUSY_HLT_alphaT_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring(
        "HLT/SUSYBSM/HLT_PFHT200_DiPFJetAve90_PFAlphaT0p51_v"
        "HLT/SUSYBSM/HLT_PFHT200_DiPFJetAve90_PFAlphaT0p57_v",
        "HLT/SUSYBSM/HLT_PFHT250_DiPFJetAve90_PFAlphaT0p55_v",
        "HLT/SUSYBSM/HLT_PFHT300_DiPFJetAve90_PFAlphaT0p53_v",
        "HLT/SUSYBSM/HLT_PFHT350_DiPFJetAve90_PFAlphaT0p52_v",
        "HLT/SUSYBSM/HLT_PFHT400_DiPFJetAve90_PFAlphaT0p51_v",
        "HLT/SUSYBSM/HLT_PFHT200_DiPFJetAve90_PFAlphaT0p63_v",
        "HLT/SUSYBSM/HLT_PFHT250_DiPFJetAve90_PFAlphaT0p58_v",
        "HLT/SUSYBSM/HLT_PFHT300_DiPFJetAve90_PFAlphaT0p54_v",
        "HLT/SUSYBSM/HLT_PFHT350_DiPFJetAve90_PFAlphaT0p53_v",
        "HLT/SUSYBSM/HLT_PFHT400_DiPFJetAve90_PFAlphaT0p52_v",
        ),
    
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfHtTurnOn_eff 'Turn-on vs PF HT; HT (GeV); #epsilon' pfHtTurnOn_num pfHtTurnOn_den",
       "pfAlphaTTurnOn_eff 'Turn-on vs PF alpha T; AlphaT (GeV); #epsilon' pfAlphaTTurnOn_num pfAlphaTTurnOn_den",
       # "caloHtTurnOn_eff 'Turn-on vs Calo HT; HT (GeV); #epsilon' caloHtTurnOn_num caloHtTurnOn_den",
       # "caloAlphaTTurnOn_eff 'Turn-on vs Calo alpha T; AlphaT (GeV); #epsilon' caloAlphaTTurnOn_num caloAlphaTTurnOn_den",
    )
)
