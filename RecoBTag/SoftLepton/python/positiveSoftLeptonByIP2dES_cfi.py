import FWCore.ParameterSet.Config as cms

positiveSoftLeptonByIP2d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("positive")
)
