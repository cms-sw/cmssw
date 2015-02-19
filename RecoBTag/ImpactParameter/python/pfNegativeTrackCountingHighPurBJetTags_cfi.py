import FWCore.ParameterSet.Config as cms

pfNegativeTrackCountingHighPurBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateNegativeTrackCounting3D3rdComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"))
)
