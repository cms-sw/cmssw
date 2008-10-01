import FWCore.ParameterSet.Config as cms

negativeSoftLeptonByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftLeptonByIP2d'),
    tagInfos = cms.VInputTag(cms.InputTag("softLeptonTagInfos"))
)
