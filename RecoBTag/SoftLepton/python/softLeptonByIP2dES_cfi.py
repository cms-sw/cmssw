import FWCore.ParameterSet.Config as cms

softLeptonByIP2d = cms.ESProducer("LeptonTaggerByIPESProducer",
    use3d = cms.bool(False),
    ipSign = cms.string("any")
)
