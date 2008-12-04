import FWCore.ParameterSet.Config as cms

skimming = cms.EDFilter("EcalSkim",
    #cosmic cluster energy threshold in GeV
    energyCutEB = cms.untracked.double(2.0),
    energyCutEE = cms.untracked.double(2.0),
    endcapClusterCollection = cms.InputTag("cosmicSuperClusters","CosmicEndcapSuperClusters"),
    barrelClusterCollection = cms.InputTag("cosmicSuperClusters","CosmicBarrelSuperClusters")
)


