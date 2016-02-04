import FWCore.ParameterSet.Config as cms

negativeSoftMuon = cms.ESProducer("MuonTaggerESProducer",
    ipSign = cms.string("negative")
)
