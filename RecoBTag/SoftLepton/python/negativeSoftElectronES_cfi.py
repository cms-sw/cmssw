import FWCore.ParameterSet.Config as cms

negativeSoftElectron = cms.ESProducer("ElectronTaggerESProducer",
    ipSign = cms.string("negative")
)
