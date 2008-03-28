import FWCore.ParameterSet.Config as cms

zToMuMuOneTrackGenParticlesMatch = cms.EDFilter("MCTruthCompositeMatcher",
    src = cms.InputTag("zToMuMuOneTrack"),
    matchMaps = cms.VInputTag(cms.InputTag("allMuonsGenParticlesMatch"), cms.InputTag("allTracksGenParticlesLeptonMatch"))
)


