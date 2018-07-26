import FWCore.ParameterSet.Config as cms

l1PhaseIITree = cms.EDAnalyzer("L1PhaseIITreeProducer",

   jetToken = cms.untracked.InputTag("simCaloStage2Digis"),
   muonToken = cms.untracked.InputTag("simGmtStage2Digis"),
   sumToken = cms.untracked.InputTag("simCaloStage2Digis"),
   tauTokens = cms.untracked.VInputTag("simCaloStage2Digis"),

   egTokens = cms.VInputTag(cms.InputTag("l1EGammaCrystalsProducer","L1EGammaCollectionBXVWithCuts"),cms.InputTag("l1EGammaEEProducer","L1EGammaCollectionBXVWithCuts")),
   tkEGTokens = cms.VInputTag( cms.InputTag("L1TkElectronsCrystal","EG"),cms.InputTag("L1TkElectronsHGC","EG") ),
   tkEMTokens = cms.VInputTag( cms.InputTag("L1TkPhotonsCrystal","EG"),cms.InputTag("L1TkPhotonsHGC","EG") ),

   tkTauToken = cms.InputTag("L1TkTauFromCalo",""), # ?
   TkGlbMuonToken = cms.InputTag("L1TkGlbMuons",""),                                            
   tkTrackerJetToken = cms.InputTag("L1TrackerJets","L1TrackerJets"),                                            
   tkCaloJetToken = cms.InputTag("L1TkCaloJets","L1TkCaloJets"),
   tkMetToken = cms.InputTag("L1TrackerEtMiss","MET"),
   tkMhtTokens = cms.VInputTag( cms.InputTag("L1TrackerHTMiss5GeV","L1TrackerHTMiss"),cms.InputTag("L1TrackerHTMiss10GeV","L1TrackerHTMiss"),cms.InputTag("L1TrackerHTMiss","L1TrackerHTMiss"),cms.InputTag("L1TrackerHTMiss20GeV","L1TrackerHTMiss"),cms.InputTag("L1TrackerHTMiss30GeV","L1TrackerHTMiss")),

   ak4L1PF = cms.InputTag("ak4L1Puppi"),
 
   muonKalman = cms.InputTag("simKBmtfDigis","BMTF"),

   l1PFMet = cms.InputTag("l1MetPuppi"),

   maxL1Extra = cms.uint32(20)
)

