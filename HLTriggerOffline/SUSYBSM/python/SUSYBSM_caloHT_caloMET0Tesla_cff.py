import FWCore.ParameterSet.Config as cms

SUSY_HLT_caloHT_caloMET0Tesla = cms.EDAnalyzer("SUSY_HLT_InclusiveCaloHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  caloMETCollection = cms.InputTag("caloMet"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_HT250_CaloMET70_JetIdCleaned_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerFilter = cms.InputTag('hltHT250', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_caloHT_caloMET0Tesla_FASTSIM = cms.EDAnalyzer("SUSY_HLT_InclusiveCaloHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  caloMETCollection = cms.InputTag("caloMet"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_HT250_CaloMET70_JetIdCleaned_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerFilter = cms.InputTag('hltHT250', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)


SUSY_HLT_caloHT_caloMET0Tesla_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
  subDirs = cms.untracked.vstring("HLT/SUSYBSM/HLT_HT250_CaloMET70_JetIdCleaned_v"),
  efficiency = cms.vstring(
    "caloMetTurnOn_eff 'Efficiency vs CaloMET' caloMetTurnOn_num caloMetTurnOn_den",
    "caloHTTurnOn_eff 'Efficiency vs CaloHT' caloHTTurnOn_num caloHTTurnOn_den"
    ),
  resolution = cms.vstring("")
)


SUSY_HLT_caloHT_caloMET0Tesla_FASTSIM_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
  subDirs = cms.untracked.vstring("HLT/SUSYBSM/HLT_HT250_CaloMET70_JetIdCleaned_v"),
  efficiency = cms.vstring(
    "caloMetTurnOn_eff 'Efficiency vs CaloMET' caloMetTurnOn_num caloMetTurnOn_den",
    "caloHTTurnOn_eff 'Efficiency vs CaloHT' caloHTTurnOn_num caloHTTurnOn_den"
    ),
  resolution = cms.vstring("")
)
