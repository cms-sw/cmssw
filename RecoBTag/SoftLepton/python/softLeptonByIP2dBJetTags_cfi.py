import FWCore.ParameterSet.Config as cms

softLeptonByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softLeptonByIP2d'),
    tagInfos = cms.VInputTag(cms.InputTag("softLeptonTagInfos"))
)
