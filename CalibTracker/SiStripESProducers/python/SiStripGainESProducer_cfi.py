import FWCore.ParameterSet.Config as cms

siStripGainESProducer = cms.ESProducer("SiStripGainESProducer",
    appendToDataLabel = cms.string(''),
    printDebug = cms.untracked.bool(False),
    NormalizationFactor = cms.double(1.0),
    AutomaticNormalization = cms.bool(False),
    APVGain = cms.VPSet(
        cms.PSet(
            Record = cms.string('SiStripApvGainRcd'),
            Label = cms.untracked.string('')
        ),
    )
)


