import FWCore.ParameterSet.Config as cms

l1JetRecoTree = cms.EDAnalyzer("L1JetRecoTreeProducer",
  pfJetToken              = cms.untracked.InputTag("ak4PFCHSJetsL1FastL2L3Residual"),
  jecToken                = cms.untracked.InputTag(""),
  maxJet                  = cms.uint32(20),
  jetptThreshold          = cms.double(10),
  pfMetToken              = cms.untracked.InputTag("pfMet")
)


