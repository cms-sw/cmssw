import FWCore.ParameterSet.Config as cms

combinedSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    ipTagInfos = cms.InputTag("impactParameterTagInfos"),
    jetTagComputer = cms.string('combinedSecondaryVertex'),
    svTagInfos = cms.InputTag("secondaryVertexTagInfos")
)


