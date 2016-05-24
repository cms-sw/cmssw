import FWCore.ParameterSet.Config as cms

SUSY_HLT_MET_MUON = cms.EDAnalyzer("SUSY_HLT_Muon_Hadronic",
  trigSummary = cms.InputTag('hltTriggerSummaryAOD','','HLT'),
  MuonCollection = cms.InputTag("muons"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_PFMET120_Mu5_v'),
  TriggerPathAuxiliaryForMuon = cms.string('HLT_PFHT900_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerFilter = cms.InputTag('hltPFMET120Mu5L3PreFiltered','','HLT'), #the last filter in the path
  ptMuonOffline = cms.untracked.double(7.0), 
  etaMuonOffline = cms.untracked.double(5.0), 
  HTOffline = cms.untracked.double(0.0),
  METOffline = cms.untracked.double(200.0),
  PtThrJet = cms.untracked.double(30.0),
  EtaThrJet = cms.untracked.double(3.0)
)


SUSY_HLT_MET_MUON_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_PFMET120_Mu5_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfHTTurnOn_eff 'Turn-on vs HT; PFHT (GeV); #epsilon' pfHTTurnOn_num pfHTTurnOn_den",
       "pfMetTurnOn_eff 'Turn-on vs MET; PFMET (GeV); #epsilon' pfMetTurnOn_num pfMetTurnOn_den",
       "MuTurnOn_eff 'Turn-on vs Mu pT; pT (GeV); #epsilon' MuTurnOn_num MuTurnOn_den",
    )
)
