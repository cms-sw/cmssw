import FWCore.ParameterSet.Config as cms

softPFMuonByIP2dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    ipSign = cms.string('any'),
    use3d = cms.bool(False)
)
