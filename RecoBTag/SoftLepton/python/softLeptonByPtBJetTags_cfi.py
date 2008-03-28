import FWCore.ParameterSet.Config as cms

softLeptonByPtBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("softLeptonTagInfos"),
    jetTagComputer = cms.string('softLeptonByPt')
)


