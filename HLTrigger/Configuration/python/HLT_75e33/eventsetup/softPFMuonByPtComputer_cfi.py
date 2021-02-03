import FWCore.ParameterSet.Config as cms

softPFMuonByPtComputer = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string('any')
)
