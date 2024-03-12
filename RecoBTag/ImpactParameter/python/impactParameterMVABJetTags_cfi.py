import FWCore.ParameterSet.Config as cms

impactParameterMVABJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('impactParameterMVAComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


# foo bar baz
# Z7IAc2a59x45m
