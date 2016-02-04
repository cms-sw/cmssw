import FWCore.ParameterSet.Config as cms

negativeSoftLeptonByPt = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("negative")
)
