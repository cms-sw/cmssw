import FWCore.ParameterSet.Config as cms

negativeSoftPFMuonByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string('negative')
)
