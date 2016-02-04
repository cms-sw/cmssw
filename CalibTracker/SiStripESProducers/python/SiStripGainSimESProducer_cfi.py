import FWCore.ParameterSet.Config as cms

siStripGainSimESProducer = cms.ESProducer("SiStripGainSimESProducer",
    appendToDataLabel = cms.string(''),
    printDebug = cms.untracked.bool(False),
    AutomaticNormalization = cms.bool(False),
    APVGain = cms.VPSet(
        cms.PSet(
            Record = cms.string('SiStripApvGainSimRcd'),
            Label = cms.untracked.string(''),
            NormalizationFactor = cms.untracked.double(1.)
        ),
    )
)
