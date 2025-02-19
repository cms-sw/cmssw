import FWCore.ParameterSet.Config as cms

allSuperClusters = cms.EDProducer("EcalCandidateProducer",
    src = cms.InputTag("hybridSuperClusters"),
    particleType = cms.string('gamma')
)


