import FWCore.ParameterSet.Config as cms

positiveSoftLeptonByIP3d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(True),
    ipSign = cms.string("positive")
)
