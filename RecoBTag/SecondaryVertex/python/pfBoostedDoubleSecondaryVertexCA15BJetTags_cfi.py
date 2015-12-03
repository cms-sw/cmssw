import FWCore.ParameterSet.Config as cms

pfBoostedDoubleSecondaryVertexCA15BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateBoostedDoubleSecondaryVertexCA15Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfosCA15"),
                             cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfosCA15"))
)
