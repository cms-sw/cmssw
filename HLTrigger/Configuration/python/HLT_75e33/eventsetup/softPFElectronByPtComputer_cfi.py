import FWCore.ParameterSet.Config as cms

softPFElectronByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string('any')
)
