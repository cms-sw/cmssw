import FWCore.ParameterSet.Config as cms

softPFElectronByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFElectronByIP2dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
softPFElectronByIP2dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("any")
)
negativeSoftPFElectronByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFElectronByIP2dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
negativeSoftPFElectronByIP2dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("negative")
)
positiveSoftPFElectronByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFElectronByIP2dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
positiveSoftPFElectronByIP2dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("positive")
)
softPFMuonByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFMuonByIP2dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
softPFMuonByIP2dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("any")
)
negativeSoftPFMuonByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFMuonByIP2dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
negativeSoftPFMuonByIP2dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("negative")
)
positiveSoftPFMuonByIP2dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFMuonByIP2dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
positiveSoftPFMuonByIP2dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("positive")
)
