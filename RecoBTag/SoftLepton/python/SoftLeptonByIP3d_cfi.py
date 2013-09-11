import FWCore.ParameterSet.Config as cms
softPFElectronByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFElectronByIP3d'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
softPFElectronByIP3d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(True),
    ipSign = cms.string("any")
)
negativeSoftPFElectronByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFElectronByIP3d'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
negativeSoftPFElectronByIP3d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(True),
    ipSign = cms.string("negative")
)
positiveSoftPFElectronByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFElectronByIP3d'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
positiveSoftPFElectronByIP3d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(True),
    ipSign = cms.string("positive")
)
softPFMuonByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFMuonByIP3d'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
softPFMuonByIP3d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(True),
    ipSign = cms.string("any")
)
negativeSoftPFMuonByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFMuonByIP3d'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
negativeSoftPFMuonByIP3d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(True),
    ipSign = cms.string("negative")
)
positiveSoftPFMuonByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFMuonByIP3d'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
positiveSoftPFMuonByIP3d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(True),
    ipSign = cms.string("positive")
)
