import FWCore.ParameterSet.Config as cms

positiveSoftMuon = cms.ESProducer("MuonTaggerESProducer",
    ipSign = cms.string("positive")
)
