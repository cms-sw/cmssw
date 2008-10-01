import FWCore.ParameterSet.Config as cms

softElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softElectron'),
    tagInfos = cms.VInputTag(cms.InputTag("softElectronTagInfos"))
)
