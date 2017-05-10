import FWCore.ParameterSet.Config as cms

SUSY_HLT_MET_BTAG = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_PFMET120_BTagCSV0p72_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v'),
  TriggerFilter = cms.InputTag('hltPFMET120Filter', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)


SUSY_HLT_MET_BTAG_FASTSIM = cms.EDAnalyzer("SUSY_HLT_InclusiveHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_PFMET120_BTagCSV0p72_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v'),
  TriggerFilter = cms.InputTag('hltPFMET120Filter', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)


SUSY_HLT_MET_BTAG_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
  subDirs = cms.untracked.vstring("HLT/SUSYBSM/HLT_PFMET120_BTagCSV0p72"),
  efficiency = cms.vstring(
    "pfMetTurnOn_eff 'Efficiency vs PFMET' pfMetTurnOn_num pfMetTurnOn_den",
    "pfHTTurnOn_eff 'Efficiency vs PFHT' pfHTTurnOn_num pfHTTurnOn_den"
    ),
  resolution = cms.vstring("")
)


SUSY_HLT_MET_BTAG_FASTSIM_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
  subDirs = cms.untracked.vstring("HLT/SUSYBSM/HLT_PFMET120_BTagCSV0p72"),
  efficiency = cms.vstring(
    "pfMetTurnOn_eff 'Efficiency vs PFMET' pfMetTurnOn_num pfMetTurnOn_den",
    "pfHTTurnOn_eff 'Efficiency vs PFHT' pfHTTurnOn_num pfHTTurnOn_den"
    ),
  resolution = cms.vstring("")
)



