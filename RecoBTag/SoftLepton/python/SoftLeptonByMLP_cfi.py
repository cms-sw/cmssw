import FWCore.ParameterSet.Config as cms
softPFElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFElectron'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
softPFElectron = cms.ESProducer("ElectronTaggerESProducer",
    ipSign = cms.string("any")
)
negativeSoftPFElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFElectron'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
negativeSoftPFElectron = cms.ESProducer("ElectronTaggerESProducer",
    ipSign = cms.string("negative")
)
positiveSoftPFElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFElectron'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
positiveSoftPFElectron = cms.ESProducer("ElectronTaggerESProducer",
    ipSign = cms.string("positive")
)
softPFMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFMuon'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
softPFMuon = cms.ESProducer("MuonTaggerESProducer",
    ipSign = cms.string("any")
)
negativeSoftPFMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFMuon'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
negativeSoftPFMuon = cms.ESProducer("MuonTaggerESProducer",
    ipSign = cms.string("negative")
)
positiveSoftPFMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFMuon'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
positiveSoftPFMuon = cms.ESProducer("MuonTaggerESProducer",
    ipSign = cms.string("positive")
)
