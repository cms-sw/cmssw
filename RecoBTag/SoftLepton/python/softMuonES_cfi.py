import FWCore.ParameterSet.Config as cms

softMuon = cms.ESProducer("MuonTaggerESProducer",
    ipSign = cms.string("any")
)
