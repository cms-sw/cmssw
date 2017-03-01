import FWCore.ParameterSet.Config as cms

pfChargeBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateChargeBTagComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
                             cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfos"),
                             cms.InputTag("softPFMuonsTagInfos"),
                             cms.InputTag("softPFElectronsTagInfos"))
)
