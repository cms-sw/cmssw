import FWCore.ParameterSet.Config as cms

allSuperClusterCandidates = cms.EDProducer("ConcreteEcalCandidateProducer",
    src = cms.InputTag("hybridSuperClusters"),
    particleType = cms.string('gamma')
)


# foo bar baz
# 3ZrjsYRqy0z8a
