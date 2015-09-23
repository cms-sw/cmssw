import FWCore.ParameterSet.Config as cms

SUSY_HLT_PhotonCaloHT = cms.EDAnalyzer("SUSY_HLT_PhotonCaloHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  photonCollection = cms.InputTag("gedPhotons"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Photon90_CaloIdL_HT300_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v'),
  TriggerFilterPhoton = cms.InputTag('hltEG90L1SingleEG40HEFilter', '', 'HLT'),
  TriggerFilterHt = cms.InputTag('hltHT300', '', 'HLT'),
  ptThrOffline = cms.untracked.double( 100 ),
  htThrOffline = cms.untracked.double( 400 ),
)

SUSY_HLT_PhotonCaloHT_FASTSIM = cms.EDAnalyzer("SUSY_HLT_PhotonCaloHT",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  photonCollection = cms.InputTag("gedPhotons"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Photon90_CaloIdL_HT300_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v'),
  TriggerFilterPhoton = cms.InputTag('hltEG90L1SingleEG40HEFilter', '', 'HLT'),
  TriggerFilterHt = cms.InputTag('hltHT300', '', 'HLT'),
  ptThrOffline = cms.untracked.double( 100 ),
  htThrOffline = cms.untracked.double( 400 ),
)

SUSY_HLT_PhotonCaloHT_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Photon90_CaloIdL_HT300_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "photonPtTurnOn_eff 'Turn-on vs photon; E_{T} (GeV); #epsilon' photonTurnOn_num photonTurnOn_den",
       "htTurnOn_eff 'Turn-on vs H_{T}; H_{T} (GeV); #epsilon' HtTurnOn_num HtTurnOn_den",
    )
)

SUSY_HLT_PhotonCaloHT_FASTSIM_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Photon90_CaloIdL_HT300_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "photonPtTurnOn_eff 'Turn-on vs photon; E_{T} (GeV); #epsilon' photonTurnOn_num photonTurnOn_den",
       "htTurnOn_eff 'Turn-on vs H_{T}; H_{T} (GeV); #epsilon' HtTurnOn_num HtTurnOn_den",
    )
)



