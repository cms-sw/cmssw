import FWCore.ParameterSet.Config as cms

positiveSoftPFElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftElectron'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
