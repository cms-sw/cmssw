import FWCore.ParameterSet.Config as cms

jetProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('jetProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"))
)


# foo bar baz
# mwx64tVzreiTj
# QdV6PI41xYxKY
