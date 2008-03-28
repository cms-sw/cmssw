import FWCore.ParameterSet.Config as cms

softLeptonByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("softLeptonTagInfos"),
    jetTagComputer = cms.string('softLeptonByIP2d')
)


