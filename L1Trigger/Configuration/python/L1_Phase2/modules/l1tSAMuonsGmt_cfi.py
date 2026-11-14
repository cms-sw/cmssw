import FWCore.ParameterSet.Config as cms

l1tSAMuonsGmt = cms.EDProducer("Phase2L1TGMTSAMuonGhostCleaner",
    barrelDisp = cms.InputTag("l1tKMTFMuonsGmt","displaced"),
    barrelPrompt = cms.InputTag("l1tKMTFMuonsGmt","prompt"),
    forwardDisp = cms.InputTag("l1tFwdMuonsGmt","displaced"),
    forwardPrompt = cms.InputTag("l1tFwdMuonsGmt","prompt")
)
