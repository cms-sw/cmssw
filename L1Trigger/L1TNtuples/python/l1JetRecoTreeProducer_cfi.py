import FWCore.ParameterSet.Config as cms

l1JetRecoTreeProducer = cms.EDAnalyzer("L1JetRecoTreeProducer",
  caloJetTag              = cms.untracked.InputTag("ak4CaloJets"),
  pfJetTag                = cms.untracked.InputTag("ak4CaloJets"),
  jetIdTag                = cms.untracked.InputTag(""),
  jetCorrToken            = cms.untracked.InputTag(""),
  maxJet                  = cms.uint32(20),
  jetptThreshold          = cms.double(10)
)

