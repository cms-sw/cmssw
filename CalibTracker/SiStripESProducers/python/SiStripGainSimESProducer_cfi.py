import FWCore.ParameterSet.Config as cms

siStripGainSimESProducer = cms.ESProducer("SiStripGainSimESProducer",
    appendToDataLabel = cms.string(''),
    printDebug = cms.untracked.bool(False),
    NormalizationFactor = cms.double(1.0),
    AutomaticNormalization = cms.bool(False),
    APVGain = cms.string('')
)


