import FWCore.ParameterSet.Config as cms

negativeSoftLeptonByIP2d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("negative")
)
