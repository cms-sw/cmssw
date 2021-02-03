import FWCore.ParameterSet.Config as cms

positiveSoftPFElectronByIP2dComputer = cms.ESProducer("LeptonTaggerByIPESProducer",
    ipSign = cms.string('positive'),
    use3d = cms.bool(False)
)
