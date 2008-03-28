import FWCore.ParameterSet.Config as cms

softLeptonByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("softLeptonTagInfos"),
    jetTagComputer = cms.string('softLeptonByIP3d')
)


