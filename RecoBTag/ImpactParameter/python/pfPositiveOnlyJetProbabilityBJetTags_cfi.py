import FWCore.ParameterSet.Config as cms

pfPositiveOnlyJetProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidatePositiveOnlyJetProbabilityComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"))
)
# foo bar baz
# bdHUS2iDDbUXx
# 7sJimtPJC2Pbn
