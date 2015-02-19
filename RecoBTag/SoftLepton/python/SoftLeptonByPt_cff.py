import FWCore.ParameterSet.Config as cms

softPFElectronByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFElectronByPtComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
softPFElectronByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("any")
)
negativeSoftPFElectronByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFElectronByPtComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
negativeSoftPFElectronByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("negative")
)
positiveSoftPFElectronByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFElectronByPtComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
positiveSoftPFElectronByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("positive")
)
softPFMuonByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFMuonByPtComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
softPFMuonByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("any")
)
negativeSoftPFMuonByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFMuonByPtComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
negativeSoftPFMuonByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("negative")
)
positiveSoftPFMuonByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFMuonByPtComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
positiveSoftPFMuonByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("positive")
)
