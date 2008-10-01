import FWCore.ParameterSet.Config as cms

negativeSoftElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftElectron'),
    tagInfos = cms.VInputTag(cms.InputTag("softElectronTagInfos"))
)
