import FWCore.ParameterSet.Config as cms

negativeSoftPFElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftElectron'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFLeptonsTagInfo","SPFElectrons"))
)
