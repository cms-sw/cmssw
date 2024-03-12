import FWCore.ParameterSet.Config as cms

pfPositiveOnlyJetBProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidatePositiveOnlyJetBProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"))
)


# foo bar baz
# tVnSwZJB81jZ6
# ZqKwk2BvEqKhf
