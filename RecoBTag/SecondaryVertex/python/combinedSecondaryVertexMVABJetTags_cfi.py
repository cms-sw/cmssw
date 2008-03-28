import FWCore.ParameterSet.Config as cms

combinedSecondaryVertexMVABJetTags = cms.EDProducer("JetTagProducer",
    ipTagInfos = cms.InputTag("impactParameterTagInfos"),
    jetTagComputer = cms.string('combinedSecondaryVertexMVA'),
    svTagInfos = cms.InputTag("secondaryVertexTagInfos")
)


