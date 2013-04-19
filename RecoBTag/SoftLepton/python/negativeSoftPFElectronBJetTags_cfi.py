import FWCore.ParameterSet.Config as cms

negativeSoftPFElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftElectron'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
