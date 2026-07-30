import FWCore.ParameterSet.Config as cms

hltEgammaElectronPixelSeedsPortable = cms.EDProducer("ElectronNHitSeedAlpakaProducer@alpaka",
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    initialSeeds = cms.InputTag("hltElePixelSeedsCombinedL1Seeded"),
    superClusters = cms.InputTag("hltEgammaSuperClustersToPixelMatchL1Seeded")
)
