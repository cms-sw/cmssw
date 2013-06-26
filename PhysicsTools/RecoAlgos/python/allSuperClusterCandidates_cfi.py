import FWCore.ParameterSet.Config as cms

allSuperClusterCandidates = cms.EDProducer("ConcreteEcalCandidateProducer",
    src = cms.InputTag("hybridSuperClusters"),
    particleType = cms.string('gamma')
)


