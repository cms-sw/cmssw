import FWCore.ParameterSet.Config as cms

softLeptonByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softLeptonByIP3d'),
    tagInfos = cms.VInputTag(cms.InputTag("softLeptonTagInfos"))
)
