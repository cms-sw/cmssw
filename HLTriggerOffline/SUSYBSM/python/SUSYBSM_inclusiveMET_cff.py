import FWCore.ParameterSet.Config as cms

SUSY_HLT_InclusiveMET = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_PFMET170_NoiseCleaned_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v'),
  TriggerFilter = cms.InputTag('hltPFMET170Filter', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_InclusiveMET_NoNoiseCleaning = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_PFMET170_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v'),
  TriggerFilter = cms.InputTag('hltPFMET170Filter', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)


SUSY_HLT_InclusiveMET_HBHECleaned = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_PFMET170_HBHECleaned_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v'),
  TriggerFilter = cms.InputTag('hltPFMET170Filter', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)


SUSY_HLT_InclusiveMET_JetIdCleaned = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_PFMET170_JetIdCleaned_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v'),
  TriggerFilter = cms.InputTag('hltPFMET170Filter', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)


SUSY_HLT_InclusiveMET_FASTSIM = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_PFMET170_NoiseCleaned_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v'),
  TriggerFilter = cms.InputTag('hltPFMET170Filter', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_InclusiveMET_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
  subDirs = cms.untracked.vstring("HLT/SUSYBSM/HLT_PFMET170_NoiseCleaned"),
  efficiency = cms.vstring(
    "pfMetTurnOn_eff 'Efficiency vs PFMET' pfMetTurnOn_num pfMetTurnOn_den",
    "pfHTTurnOn_eff 'Efficiency vs PFHT' pfHTTurnOn_num pfHTTurnOn_den"
    ),
  resolution = cms.vstring("")
)

SUSY_HLT_InclusiveMET_NoNoiseCleaning_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
  subDirs = cms.untracked.vstring("HLT/SUSYBSM/HLT_PFMET170"),
  efficiency = cms.vstring(
    "pfMetTurnOn_eff 'Efficiency vs PFMET' pfMetTurnOn_num pfMetTurnOn_den",
    "pfHTTurnOn_eff 'Efficiency vs PFHT' pfHTTurnOn_num pfHTTurnOn_den"
    ),
  resolution = cms.vstring("")
)

SUSY_HLT_InclusiveMET_HBHECleaned_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
  subDirs = cms.untracked.vstring("HLT/SUSYBSM/HLT_PFMET170_HBHECleaned"),
  efficiency = cms.vstring(
    "pfMetTurnOn_eff 'Efficiency vs PFMET' pfMetTurnOn_num pfMetTurnOn_den",
    "pfHTTurnOn_eff 'Efficiency vs PFHT' pfHTTurnOn_num pfHTTurnOn_den"
    ),
  resolution = cms.vstring("")
)

SUSY_HLT_InclusiveMET_JetIdCleaned_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
  subDirs = cms.untracked.vstring("HLT/SUSYBSM/HLT_PFMET170_JetIdCleaned"),
  efficiency = cms.vstring(
    "pfMetTurnOn_eff 'Efficiency vs PFMET' pfMetTurnOn_num pfMetTurnOn_den",
    "pfHTTurnOn_eff 'Efficiency vs PFHT' pfHTTurnOn_num pfHTTurnOn_den"
    ),
  resolution = cms.vstring("")
)


SUSY_HLT_InclusiveMET_FASTSIM_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
  subDirs = cms.untracked.vstring("HLT/SUSYBSM/HLT_PFMET170_NoiseCleaned"),
  efficiency = cms.vstring(
    "pfMetTurnOn_eff 'Efficiency vs PFMET' pfMetTurnOn_num pfMetTurnOn_den",
    "pfHTTurnOn_eff 'Efficiency vs PFHT' pfHTTurnOn_num pfHTTurnOn_den"
    ),
  resolution = cms.vstring("")
)


