import FWCore.ParameterSet.Config as cms

l1JetRecoTree = cms.EDAnalyzer("L1JetRecoTreeProducer",
  pfJetTag                = cms.untracked.InputTag("ak4PFJetsCHS"),
  jetIdTag                = cms.untracked.InputTag(""),
  jetCorrToken            = cms.untracked.InputTag(""),
  maxJet                  = cms.uint32(20),
  jetptThreshold          = cms.double(10)
)

