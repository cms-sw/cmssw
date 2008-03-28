import FWCore.ParameterSet.Config as cms

zToMuMuGenParticlesMatch = cms.EDFilter("MCTruthCompositeMatcher",
    src = cms.InputTag("zToMuMu"),
    matchMaps = cms.VInputTag(cms.InputTag("allMuonsGenParticlesMatch"))
)


