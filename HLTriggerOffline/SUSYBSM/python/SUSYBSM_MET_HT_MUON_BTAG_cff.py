import FWCore.ParameterSet.Config as cms

SUSY_HLT_MET_HT_MUON_BTAG = cms.EDAnalyzer("SUSY_HLT_Muon_Hadronic",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  MuonCollection = cms.InputTag("muons"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Mu6_PFHT200_PFMET80_BTagCSV0p72_v'),
  TriggerPathAuxiliaryForMuon = cms.string('HLT_PFHT900_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v'),
  TriggerFilter = cms.InputTag('hltMu6PFHT200PFMET125L3PreFiltered', '', 'HLT'), #the last filter in the path
  ptMuonOffline = cms.untracked.double(8.0), 
  etaMuonOffline = cms.untracked.double(5.0), 
  HTOffline = cms.untracked.double(350.0),
  METOffline = cms.untracked.double(200.0),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)


SUSY_HLT_MET_HT_MUON_BTAG_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Muon_Hadronic",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  MuonCollection = cms.InputTag("muons"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  #TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Mu6_PFHT200_PFMET80_BTagCSV0p72_v'),   
  TriggerPathAuxiliaryForMuon = cms.string('HLT_PFHT900_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_IterTrk02_v'),
  TriggerFilter = cms.InputTag('hltMu6PFHT200PFMET125L3PreFiltered', '', 'HLT'), #the last filter in the path
  ptMuonOffline = cms.untracked.double(8.0), 
  etaMuonOffline = cms.untracked.double(5.0), 
  HTOffline = cms.untracked.double(350.0),
  METOffline = cms.untracked.double(250.0),
  PtThrJet = cms.untracked.double(40.0),
  EtaThrJet = cms.untracked.double(3.0)
)


SUSY_HLT_MET_HT_MUON_BTAG_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Mu6_PFHT200_PFMET80_BTagCSV0p72"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfHTTurnOn_eff 'Turn-on vs HT; PFHT (GeV); #epsilon' pfHTTurnOn_num pfHTTurnOn_den",
       "pfMetTurnOn_eff 'Turn-on vs MET; PFMET (GeV); #epsilon' pfMetTurnOn_num pfMetTurnOn_den",
       "MuTurnOn_eff 'Turn-on vs Mu pT; pT (GeV); #epsilon' MuTurnOn_num MuTurnOn_den",
    )
)


SUSY_HLT_MET_HT_MUON_BTAG_FASTSIM_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Mu6_PFHT200_PFMET80_BTagCSV0p72"), 
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfHTTurnOn_eff 'Turn-on vs HT; PFHT (GeV); #epsilon' pfHTTurnOn_num pfHTTurnOn_den",
       "pfMetTurnOn_eff 'Turn-on vs MET; PFMET (GeV); #epsilon' pfMetTurnOn_num pfMetTurnOn_den",
       "MuTurnOn_eff 'Turn-on vs Mu pT; pT (GeV); #epsilon' MuTurnOn_num MuTurnOn_den",
    )
)




