import FWCore.ParameterSet.Config as cms

SUSY_HLT_PhotonHT = cms.EDAnalyzer("SUSY_HLT_PhotonHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLTX'),
  pfMETCollection = cms.InputTag("pfMet"),
  photonCollection = cms.InputTag("gedPhotons"),
  TriggerResults = cms.InputTag('TriggerResults','','HLTX'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Photon90_CaloId_PFHT500_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v1'),
  TriggerFilterPhoton = cms.InputTag('hltEG90HEFilter', '', 'HLTX'),
  TriggerFilterHt = cms.InputTag('hltPFHT500', '', 'HLTX'),
  ptThrOffline = cms.untracked.double( 100 ),
  htThrOffline = cms.untracked.double( 600 ),
)

SUSY_HLT_PhotonHT_FASTSIM = cms.EDAnalyzer("SUSY_HLT_PhotonHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLTX'),
  pfMETCollection = cms.InputTag("pfMet"),
  photonCollection = cms.InputTag("gedPhotons"),
  TriggerResults = cms.InputTag('TriggerResults','','HLTX'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Photon90_CaloId_PFHT500_v1'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v1'),
  TriggerFilterPhoton = cms.InputTag('hltEG90HEFilter', '', 'HLTX'),
  TriggerFilterHt = cms.InputTag('hltPFHT500', '', 'HLTX'),
  ptThrOffline = cms.untracked.double( 100 ),
  htThrOffline = cms.untracked.double( 600 ),
)

SUSY_HLT_PhotonHT_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Photon90_CaloId_PFHT500_v1"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "photonPtTurnOn_eff 'Turn-on vs photon; E_{T} (GeV); #epsilon' photonTurnOn_num photonTurnOn_den",
       "htTurnOn_eff 'Turn-on vs H_{T}; H_{T} (GeV); #epsilon' pfHtTurnOn_num pfHtTurnOn_den",
    )
)

SUSY_HLT_PhotonHT_FASTSIM_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Photon90_CaloId_PFHT500_v1"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "photonPtTurnOn_eff 'Turn-on vs photon; E_{T} (GeV); #epsilon' photonTurnOn_num photonTurnOn_den",
       "htTurnOn_eff 'Turn-on vs H_{T}; H_{T} (GeV); #epsilon' pfHtTurnOn_num pfHtTurnOn_den",
    )
)



