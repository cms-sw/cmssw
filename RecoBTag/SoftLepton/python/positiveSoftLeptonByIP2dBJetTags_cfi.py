import FWCore.ParameterSet.Config as cms

positiveSoftLeptonByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftLeptonByIP2d'),
    tagInfos = cms.VInputTag(cms.InputTag("softLeptonTagInfos"))
)
