import FWCore.ParameterSet.Config as cms

softElectron = cms.ESProducer("ElectronTaggerESProducer",
    ipSign = cms.string("any")
)
