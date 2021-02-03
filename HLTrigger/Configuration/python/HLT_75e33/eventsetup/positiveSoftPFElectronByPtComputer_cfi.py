import FWCore.ParameterSet.Config as cms

positiveSoftPFElectronByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string('positive')
)
