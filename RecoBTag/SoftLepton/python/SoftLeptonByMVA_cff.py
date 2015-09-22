import FWCore.ParameterSet.Config as cms

softPFElectronCommon = cms.PSet(
    useCondDB = cms.bool(False),
    gbrForestLabel = cms.string("btag_SoftPFElectron_TMVA420_BDT_74X_v1"),
    weightFile = cms.FileInPath('RecoBTag/SoftLepton/data/SoftPFElectron_BDT.weights.xml.gz'),
    useGBRForest = cms.bool(True),
    useAdaBoost = cms.bool(False)
)

softPFMuonCommon = cms.PSet(
    useCondDB = cms.bool(False),
    gbrForestLabel = cms.string("btag_SoftPFMuon_TMVA420_BDT_74X_v1"),
    weightFile = cms.FileInPath('RecoBTag/SoftLepton/data/SoftPFMuon_BDT.weights.xml.gz'),
    useGBRForest = cms.bool(True),
    useAdaBoost = cms.bool(True)
)

softPFElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFElectronComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
softPFElectronComputer = cms.ESProducer("ElectronTaggerESProducer",
    softPFElectronCommon,
    ipSign = cms.string("any"),
)
negativeSoftPFElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFElectronComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
negativeSoftPFElectronComputer = cms.ESProducer("ElectronTaggerESProducer",
    softPFElectronCommon,
    ipSign = cms.string("negative")
)
positiveSoftPFElectronBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFElectronComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFElectronsTagInfos"))
)
positiveSoftPFElectronComputer = cms.ESProducer("ElectronTaggerESProducer",
    softPFElectronCommon,
    ipSign = cms.string("positive")
)
softPFMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softPFMuonComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
softPFMuonComputer = cms.ESProducer("MuonTaggerESProducer",
    softPFMuonCommon,
    ipSign = cms.string("any")
)
negativeSoftPFMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftPFMuonComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
negativeSoftPFMuonComputer = cms.ESProducer("MuonTaggerESProducer",
    softPFMuonCommon,
    ipSign = cms.string("negative")
)
positiveSoftPFMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftPFMuonComputer'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
positiveSoftPFMuonComputer = cms.ESProducer("MuonTaggerESProducer",
    softPFMuonCommon,
    ipSign = cms.string("positive")
)
