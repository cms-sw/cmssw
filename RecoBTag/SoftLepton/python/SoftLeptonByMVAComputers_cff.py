import FWCore.ParameterSet.Config as cms

softPFElectronCommon = cms.PSet(
    useCondDB = cms.bool(True),
    gbrForestLabel = cms.string("btag_SoftPFElectron_BDT"),
    weightFile = cms.FileInPath('RecoBTag/SoftLepton/data/SoftPFElectron_BDT.weights.xml.gz'),
    useGBRForest = cms.bool(True),
    useAdaBoost = cms.bool(False)
)

softPFMuonCommon = cms.PSet(
    useCondDB = cms.bool(True),
    gbrForestLabel = cms.string("btag_SoftPFMuon_BDT"),
    weightFile = cms.FileInPath('RecoBTag/SoftLepton/data/SoftPFMuon_BDT.weights.xml.gz'),
    useGBRForest = cms.bool(True),
    useAdaBoost = cms.bool(True)
)

softPFElectronComputer = cms.ESProducer("ElectronTaggerESProducer",
    softPFElectronCommon,
    ipSign = cms.string("any"),
)

negativeSoftPFElectronComputer = cms.ESProducer("ElectronTaggerESProducer",
    softPFElectronCommon,
    ipSign = cms.string("negative")
)

positiveSoftPFElectronComputer = cms.ESProducer("ElectronTaggerESProducer",
    softPFElectronCommon,
    ipSign = cms.string("positive")
)

softPFMuonComputer = cms.ESProducer("MuonTaggerESProducer",
    softPFMuonCommon,
    ipSign = cms.string("any")
)

negativeSoftPFMuonComputer = cms.ESProducer("MuonTaggerESProducer",
    softPFMuonCommon,
    ipSign = cms.string("negative")
)

positiveSoftPFMuonComputer = cms.ESProducer("MuonTaggerESProducer",
    softPFMuonCommon,
    ipSign = cms.string("positive")
)
