import FWCore.ParameterSet.Config as cms

positiveSoftMuonNoIP = cms.ESProducer("MuonTaggerNoIPESProducer",
    ipSign = cms.string("positive")
)
