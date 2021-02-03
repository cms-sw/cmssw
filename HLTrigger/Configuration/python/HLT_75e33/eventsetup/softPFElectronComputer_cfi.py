import FWCore.ParameterSet.Config as cms

softPFElectronComputer = cms.ESProducer("ElectronTaggerESProducer",
    gbrForestLabel = cms.string('btag_SoftPFElectron_BDT'),
    ipSign = cms.string('any'),
    useAdaBoost = cms.bool(False),
    useCondDB = cms.bool(True),
    useGBRForest = cms.bool(True),
    weightFile = cms.FileInPath('RecoBTag/SoftLepton/data/SoftPFElectron_BDT.weights.xml.gz')
)
