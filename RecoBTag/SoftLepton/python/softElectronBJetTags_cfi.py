import FWCore.ParameterSet.Config as cms

softElectronBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("softElectronTagInfos"),
    jetTagComputer = cms.string('softElectron')
)


