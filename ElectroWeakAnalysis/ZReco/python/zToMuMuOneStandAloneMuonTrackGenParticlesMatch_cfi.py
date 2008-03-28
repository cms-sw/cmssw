import FWCore.ParameterSet.Config as cms

zToMuMuOneStandAloneMuonTrackGenParticlesMatch = cms.EDFilter("MCTruthCompositeMatcher",
    src = cms.InputTag("zToMuMuOneStandAloneMuonTrack"),
    matchMaps = cms.VInputTag(cms.InputTag("allMuonsGenParticlesMatch"), cms.InputTag("allStandAloneMuonTracksGenParticlesLeptonMatch"))
)


