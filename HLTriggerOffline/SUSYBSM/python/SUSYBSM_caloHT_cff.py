import FWCore.ParameterSet.Config as cms

SUSY_HLT_CaloHT200 = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak5PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'),
  TriggerPath = cms.string('HLT_HT200_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT200', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_CaloHT250 = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak5PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'),
  TriggerPath = cms.string('HLT_HT250_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT250', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_CaloHT300 = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak5PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'),
  TriggerPath = cms.string('HLT_HT300_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT300', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_CaloHT350 = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak5PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'),
  TriggerPath = cms.string('HLT_HT350_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT350', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_CaloHT400 = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak5PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'),
  TriggerPath = cms.string('HLT_HT400_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT400', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_CaloHT200_FASTSIM = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak5PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'),
  TriggerPath = cms.string('HLT_HT200_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT200', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_CaloHT250_FASTSIM = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak5PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'),
  TriggerPath = cms.string('HLT_HT250_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT250', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_CaloHT300_FASTSIM = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak5PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'),
  TriggerPath = cms.string('HLT_HT300_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT300', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_CaloHT350_FASTSIM = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak5PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'),
  TriggerPath = cms.string('HLT_HT350_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT350', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_CaloHT400_FASTSIM = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak5PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak5CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','TEST'),
  TriggerPath = cms.string('HLT_HT400_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v1'),
  TriggerFilter = cms.InputTag('hltHT400', '', 'TEST'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)



SUSY_HLT_CaloHT_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
  subDirs = cms.untracked.vstring(
  "HLT/SUSYBSM/HLT_HT200_v1",
  "HLT/SUSYBSM/HLT_HT250_v1",
  "HLT/SUSYBSM/HLT_HT300_v1",
  "HLT/SUSYBSM/HLT_HT350_v1",
  "HLT/SUSYBSM/HLT_HT400_v1",
  ),
  efficiency = cms.vstring(
  "pfMetTurnOn_eff 'Efficiency vs PFMET' pfMetTurnOn_num pfMetTurnOn_den",
  "pfHTTurnOn_eff 'Efficiency vs PFHT' pfHTTurnOn_num pfHTTurnOn_den"
  ),
  resolution = cms.vstring("")
)

SUSY_HLT_CaloHT_FASTSIM_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
  subDirs = cms.untracked.vstring(
  "HLT/SUSYBSM/HLT_HT200_v1",
  "HLT/SUSYBSM/HLT_HT250_v1",
  "HLT/SUSYBSM/HLT_HT300_v1",
  "HLT/SUSYBSM/HLT_HT350_v1",
  "HLT/SUSYBSM/HLT_HT400_v1",
  ),
  efficiency = cms.vstring(
  "pfMetTurnOn_eff 'Efficiency vs PFMET' pfMetTurnOn_num pfMetTurnOn_den",
  "pfHTTurnOn_eff 'Efficiency vs PFHT' pfHTTurnOn_num pfHTTurnOn_den"
  ),
  resolution = cms.vstring("")
)

