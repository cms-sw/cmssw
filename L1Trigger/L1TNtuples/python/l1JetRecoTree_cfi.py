import FWCore.ParameterSet.Config as cms

l1JetRecoTree = cms.EDAnalyzer("L1JetRecoTreeProducer",
  pfJetToken              = cms.untracked.InputTag("ak4PFJetsCHS"),
  pfJECToken              = cms.untracked.InputTag("ak4PFCHSL1FastL2L3ResidualCorrector"),
  puppiJetToken           = cms.InputTag("ak4PFJetsPuppi"),
  corrPuppiJetToken       = cms.untracked.InputTag("patJetsCorrectedPuppiJets"),
  caloJetToken            = cms.untracked.InputTag("ak4CaloJets"),
  caloJECToken            = cms.untracked.InputTag("ak4CaloL1FastL2L3ResidualCorrector"),
  caloJetIDToken          = cms.untracked.InputTag("ak4JetID"),
  maxJet                  = cms.uint32(20),
  jetptThreshold          = cms.double(30),
  jetetaMax               = cms.double(2.5),
  pfMetToken              = cms.untracked.InputTag("pfMetT1"),
  puppiMetToken           = cms.untracked.InputTag("pfMetPuppi"),
  caloMetToken            = cms.untracked.InputTag("caloMet"),
  caloMetBEToken          = cms.untracked.InputTag("caloMetBE"),
  muonToken               = cms.untracked.InputTag("muons")
)


