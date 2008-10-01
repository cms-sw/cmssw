import FWCore.ParameterSet.Config as cms

positiveSoftElectron = cms.ESProducer("ElectronTaggerESProducer",
    ipSign = cms.string("positive")
)
