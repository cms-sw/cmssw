import FWCore.ParameterSet.Config as cms
softPFElectronByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFElectronByIP2d'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
softPFElectronByIP2d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("any")
)
negativeSoftPFElectronByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFElectronByIP2d'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
negativeSoftPFElectronByIP2d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("negative")
)
positiveSoftPFElectronByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFElectronByIP2d'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
positiveSoftPFElectronByIP2d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("positive")
)
softPFMuonByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFMuonByIP2d'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
softPFMuonByIP2d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("any")
)
negativeSoftPFMuonByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFMuonByIP2d'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
negativeSoftPFMuonByIP2d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("negative")
)
positiveSoftPFMuonByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFMuonByIP2d'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
positiveSoftPFMuonByIP2d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("positive")
)
