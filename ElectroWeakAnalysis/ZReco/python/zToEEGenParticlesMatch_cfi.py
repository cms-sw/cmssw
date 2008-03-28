import FWCore.ParameterSet.Config as cms

zToEEGenParticlesMatch = cms.EDFilter("MCTruthCompositeMatcher",
    src = cms.InputTag("zToEE"),
    matchMaps = cms.VInputTag(cms.InputTag("allElectronsGenParticlesMatch"))
)


