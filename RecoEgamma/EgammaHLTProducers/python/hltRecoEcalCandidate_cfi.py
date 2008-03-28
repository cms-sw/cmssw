import FWCore.ParameterSet.Config as cms

hltRecoEcalCandidate = cms.EDFilter("EgammaHLTRecoEcalCandidateProducers",
    scHybridBarrelProducer = cms.InputTag("correctedHybridSuperClusters"),
    scIslandEndcapProducer = cms.InputTag("correctedEndcapSuperClustersWithPreshower"),
    recoEcalCandidateCollection = cms.string('')
)


