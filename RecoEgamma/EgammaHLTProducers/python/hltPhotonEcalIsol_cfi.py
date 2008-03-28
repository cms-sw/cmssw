import FWCore.ParameterSet.Config as cms

hltPhotonEcalIsol = cms.EDFilter("EgammaHLTEcalIsolationProducersRegional",
    egEcalIsoEtMin = cms.double(0.0),
    scIslandBarrelProducer = cms.InputTag("correctedIslandBarrelSuperClusters"),
    bcEndcapProducer = cms.InputTag("islandBasicClusters","islandEndcapBasicClusters"),
    bcBarrelProducer = cms.InputTag("islandBasicClusters","islandBarrelBasicClusters"),
    scIslandEndcapProducer = cms.InputTag("correctedEndcapSuperClustersWithPreshower"),
    egEcalIsoConeSize = cms.double(0.3),
    recoEcalCandidateProducer = cms.InputTag("hltRecoEcalCandidate")
)


