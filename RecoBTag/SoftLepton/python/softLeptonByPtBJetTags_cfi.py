import FWCore.ParameterSet.Config as cms

softLeptonByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softLeptonByPt'),
    tagInfos = cms.VInputTag(cms.InputTag("softLeptonTagInfos"))
)
