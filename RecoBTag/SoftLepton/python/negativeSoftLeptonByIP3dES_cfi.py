import FWCore.ParameterSet.Config as cms

negativeSoftLeptonByIP3d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(True),
    ipSign = cms.string("negative")
)
