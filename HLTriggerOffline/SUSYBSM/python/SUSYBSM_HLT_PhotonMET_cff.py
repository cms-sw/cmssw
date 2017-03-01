import FWCore.ParameterSet.Config as cms

SUSY_HLT_PhotonMET_pt36 = cms.EDAnalyzer("SUSY_HLT_PhotonMET",
   trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'),
   pfMETCollection = cms.InputTag("pfMet"),
   photonCollection = cms.InputTag("gedPhotons"),
   TriggerResults = cms.InputTag('TriggerResults','','HLT'),
   HLTProcess = cms.string('HLT'),
   TriggerPath = cms.string('HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_PFMET40_v'),
   TriggerPathBase = cms.string('HLT_IsoMu27_v'),
   ptThrOffline = cms.untracked.double(50),
   metThrOffline = cms.untracked.double(100),
)

SUSY_HLT_PhotonMET_pt50 = cms.EDAnalyzer("SUSY_HLT_PhotonMET",
   trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'),
   pfMETCollection = cms.InputTag("pfMet"),
   photonCollection = cms.InputTag("gedPhotons"),
   TriggerResults = cms.InputTag('TriggerResults','','HLT'),
   HLTProcess = cms.string('HLT'),
   TriggerPath = cms.string('HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_PFMET40_v'),
   TriggerPathBase = cms.string('HLT_IsoMu27_v'),
   ptThrOffline = cms.untracked.double(75),
   metThrOffline = cms.untracked.double(100),
)

SUSY_HLT_PhotonMET_pt75 = cms.EDAnalyzer("SUSY_HLT_PhotonMET",
   trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'),
   pfMETCollection = cms.InputTag("pfMet"),
   photonCollection = cms.InputTag("gedPhotons"),
   TriggerResults = cms.InputTag('TriggerResults','','HLT'),
   HLTProcess = cms.string('HLT'),
   TriggerPath = cms.string('HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_PFMET40_v'),
   TriggerPathBase = cms.string('HLT_IsoMu27_v'),
   ptThrOffline = cms.untracked.double(100),
   metThrOffline = cms.untracked.double(100),
)

SUSY_HLT_PhotonMET_pt36_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
   subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_PFMET40_v"),
   verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
   resolution     = cms.vstring(""),
   efficiency     = cms.vstring(
      "photonPtTurnOn_eff 'Turn-on vs photon; p_{T} (GeV); #epsilon' photonTurnOn_num photonTurnOn_den",
      "metTurnOn_eff 'Turn-on vs E_{T}^{miss}; E_{T}^{miss} (GeV); #epsilon' pfMetTurnOn_num pfMetTurnOn_den",
   )
)

SUSY_HLT_PhotonMET_pt50_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
   subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_PFMET40_v"),
   verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
   resolution     = cms.vstring(""),
   efficiency     = cms.vstring(
      "photonPtTurnOn_eff 'Turn-on vs photon; p_{T} (GeV); #epsilon' photonTurnOn_num photonTurnOn_den",
      "metTurnOn_eff 'Turn-on vs E_{T}^{miss}; E_{T}^{miss} (GeV); #epsilon' pfMetTurnOn_num pfMetTurnOn_den",
   )
)

SUSY_HLT_PhotonMET_pt75_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
   subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_PFMET40_v"),
   verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
   resolution     = cms.vstring(""),
   efficiency     = cms.vstring(
      "photonPtTurnOn_eff 'Turn-on vs photon; p_{T} (GeV); #epsilon' photonTurnOn_num photonTurnOn_den",
      "metTurnOn_eff 'Turn-on vs E_{T}^{miss}; E_{T}^{miss} (GeV); #epsilon' pfMetTurnOn_num pfMetTurnOn_den",
   )
)
