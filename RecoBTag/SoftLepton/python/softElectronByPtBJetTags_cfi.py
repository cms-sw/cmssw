import FWCore.ParameterSet.Config as cms

softElectronByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softLeptonByPt'),
    tagInfos = cms.VInputTag(cms.InputTag("softElectronTagInfos"))
)
