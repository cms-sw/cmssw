import FWCore.ParameterSet.Config as cms

pfBoostedDoubleSecondaryVertexCA15BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateBoostedDoubleSecondaryVertexCA15Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterCA15TagInfos"),
                             cms.InputTag("pfInclusiveSecondaryVertexFinderCA15TagInfos"))
)
