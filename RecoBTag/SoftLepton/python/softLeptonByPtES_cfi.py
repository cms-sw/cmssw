import FWCore.ParameterSet.Config as cms

softLeptonByPt = cms.ESProducer("LeptonTaggerByPtESProducer",
    ipSign = cms.string("any")
)
