import FWCore.ParameterSet.Config as cms

softMuonNoIP = cms.ESProducer("MuonTaggerNoIPESProducer",
    ipSign = cms.string("any")
)
