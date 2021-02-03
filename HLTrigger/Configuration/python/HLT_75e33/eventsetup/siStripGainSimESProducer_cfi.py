import FWCore.ParameterSet.Config as cms

siStripGainSimESProducer = cms.ESProducer("SiStripGainSimESProducer",
    APVGain = cms.VPSet(cms.PSet(
        Label = cms.untracked.string(''),
        NormalizationFactor = cms.untracked.double(1.0),
        Record = cms.string('SiStripApvGainSimRcd')
    )),
    AutomaticNormalization = cms.bool(False),
    appendToDataLabel = cms.string(''),
    printDebug = cms.untracked.bool(False)
)
