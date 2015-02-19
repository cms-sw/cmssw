import FWCore.ParameterSet.Config as cms

softPFElectronByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFElectronByIP3dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
softPFElectronByIP3dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(True),
    ipSign = cms.string("any")
)
negativeSoftPFElectronByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFElectronByIP3dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
negativeSoftPFElectronByIP3dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(True),
    ipSign = cms.string("negative")
)
positiveSoftPFElectronByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFElectronByIP3dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
positiveSoftPFElectronByIP3dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(True),
    ipSign = cms.string("positive")
)
softPFMuonByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFMuonByIP3dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
softPFMuonByIP3dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(True),
    ipSign = cms.string("any")
)
negativeSoftPFMuonByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFMuonByIP3dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
negativeSoftPFMuonByIP3dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(True),
    ipSign = cms.string("negative")
)
positiveSoftPFMuonByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFMuonByIP3dComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
positiveSoftPFMuonByIP3dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(True),
    ipSign = cms.string("positive")
)
