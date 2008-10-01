import FWCore.ParameterSet.Config as cms

positiveSoftElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftElectron'),
    tagInfos = cms.VInputTag(cms.InputTag("softElectronTagInfos"))
)
