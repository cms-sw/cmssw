import FWCore.ParameterSet.Config as cms

pfJetBProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateJetBProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"))
)


# foo bar baz
# RBstacQ5QVczd
# y9xTeMIPz13I7
