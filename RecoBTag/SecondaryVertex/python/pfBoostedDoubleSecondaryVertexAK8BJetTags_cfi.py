import FWCore.ParameterSet.Config as cms

pfBoostedDoubleSecondaryVertexAK8BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateBoostedDoubleSecondaryVertexAK8Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfosAK8"),
                             cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfosAK8"),
                             cms.InputTag("softPFMuonsTagInfosAK8"),
                             cms.InputTag("softPFElectronsTagInfosAK8"))
)
