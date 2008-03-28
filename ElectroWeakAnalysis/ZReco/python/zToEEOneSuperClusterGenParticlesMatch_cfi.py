import FWCore.ParameterSet.Config as cms

zToEEOneSuperClusterGenParticlesMatch = cms.EDFilter("MCTruthCompositeMatcher",
    src = cms.InputTag("zToEEOneSuperCluster"),
    matchMaps = cms.VInputTag(cms.InputTag("allElectronsGenParticlesMatch"), cms.InputTag("allSuperClustersGenParticlesLeptonMatch"))
)


