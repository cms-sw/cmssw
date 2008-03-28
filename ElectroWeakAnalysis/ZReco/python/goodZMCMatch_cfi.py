import FWCore.ParameterSet.Config as cms

goodZMCMatch = cms.EDFilter("GenParticleMatchMerger",
    src = cms.VInputTag(cms.InputTag("goodMuonMCMatch"), cms.InputTag("goodTrackMCMatch"), cms.InputTag("goodStandAloneMuonTrackMCMatch"), cms.InputTag("goodZToMuMuMCMatch"), cms.InputTag("goodZToMuMuOneTrackMCMatch"), cms.InputTag("goodZToMuMuOneStandAloneMuonTrackMCMatch"))
)


