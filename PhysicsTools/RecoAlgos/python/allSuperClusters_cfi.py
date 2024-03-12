import FWCore.ParameterSet.Config as cms

allSuperClusters = cms.EDProducer("EcalCandidateProducer",
    src = cms.InputTag("hybridSuperClusters"),
    particleType = cms.string('gamma')
)


# foo bar baz
# YXLNM5iru6gDT
# IKox0zIhsBUH8
