import FWCore.ParameterSet.Config as cms

positiveSoftLeptonByPt = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("positive")
)
