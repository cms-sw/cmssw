import FWCore.ParameterSet.Config as cms

hltRecoEcalCandidate = cms.EDProducer("EgammaHLTRecoEcalCandidateProducers",
    scHybridBarrelProducer = cms.InputTag("correctedHybridSuperClusters"),
    scIslandEndcapProducer = cms.InputTag("correctedEndcapSuperClustersWithPreshower"),
    recoEcalCandidateCollection = cms.string('')
)


