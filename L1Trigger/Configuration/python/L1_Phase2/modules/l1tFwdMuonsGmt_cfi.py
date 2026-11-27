import FWCore.ParameterSet.Config as cms

l1tFwdMuonsGmt = cms.EDProducer("Phase2L1TGMTFwdMuonTranslator",
    emtfTracks = cms.InputTag("simEmtfDigisPhase2"),
    omtfConstrSAMs = cms.InputTag("simOmtfPhase2Digis","constrCands"),
    omtfUnconstrSAMs = cms.InputTag("simOmtfPhase2Digis","unconstrCands"),
    stubs = cms.InputTag("l1tStubsGmt","tps")
)
