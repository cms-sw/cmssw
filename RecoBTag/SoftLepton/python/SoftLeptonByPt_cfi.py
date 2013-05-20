import FWCore.ParameterSet.Config as cms
softPFElectronByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFElectronByPt'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
softPFElectronByPt = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("any")
)
negativeSoftPFElectronByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFElectronByPt'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
negativeSoftPFElectronByPt = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("negative")
)
positiveSoftPFElectronByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFElectronByPt'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
positiveSoftPFElectronByPt = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("positive")
)
softPFMuonByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFMuonByPt'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
softPFMuonByPt = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("any")
)
negativeSoftPFMuonByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFMuonByPt'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
negativeSoftPFMuonByPt = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("negative")
)
positiveSoftPFMuonByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFMuonByPt'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
positiveSoftPFMuonByPt = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("positive")
)
