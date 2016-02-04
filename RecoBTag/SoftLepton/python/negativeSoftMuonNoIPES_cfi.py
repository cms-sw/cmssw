import FWCore.ParameterSet.Config as cms

negativeSoftMuonNoIP = cms.ESProducer("MuonTaggerNoIPESProducer",
    ipSign = cms.string("negative")
)
