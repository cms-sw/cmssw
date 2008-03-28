import FWCore.ParameterSet.Config as cms

zToEEOneTrackGenParticlesMatch = cms.EDFilter("MCTruthCompositeMatcher",
    src = cms.InputTag("zToEEOneTrack"),
    matchMaps = cms.VInputTag(cms.InputTag("allElectronsGenParticlesMatch"), cms.InputTag("allTracksGenParticlesLeptonMatch"))
)


