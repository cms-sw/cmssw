import FWCore.ParameterSet.Config as cms

jetBProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('jetBProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


# foo bar baz
# GzXAvsQt1Z2BS
