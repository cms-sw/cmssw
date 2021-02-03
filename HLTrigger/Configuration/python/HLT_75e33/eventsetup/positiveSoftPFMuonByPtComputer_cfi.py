import FWCore.ParameterSet.Config as cms

positiveSoftPFMuonByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string('positive')
)
