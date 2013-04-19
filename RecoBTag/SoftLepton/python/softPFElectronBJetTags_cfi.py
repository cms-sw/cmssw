import FWCore.ParameterSet.Config as cms

softPFElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softElectron'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
