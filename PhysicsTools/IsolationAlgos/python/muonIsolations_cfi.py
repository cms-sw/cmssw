import FWCore.ParameterSet.Config as cms

muonIsolations = cms.EDFilter("ValeMapFloatMerger",
    src = cms.VInputTag(cms.InputTag("goodMuonIsolations"), cms.InputTag("goodTrackIsolations"), cms.InputTag("goodStandAloneMuonTrackIsolations"))
)


