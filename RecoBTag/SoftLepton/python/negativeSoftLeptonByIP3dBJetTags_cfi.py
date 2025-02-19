import FWCore.ParameterSet.Config as cms

negativeSoftLeptonByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftLeptonByIP3d'),
    tagInfos = cms.VInputTag(cms.InputTag("softLeptonTagInfos"))
)
