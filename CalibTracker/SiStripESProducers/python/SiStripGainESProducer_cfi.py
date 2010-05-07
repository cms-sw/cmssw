import FWCore.ParameterSet.Config as cms

siStripGainESProducer = cms.ESProducer("SiStripGainESProducer",
    appendToDataLabel = cms.string(''),
    printDebug = cms.untracked.bool(False),
    AutomaticNormalization = cms.bool(False),
    APVGain = cms.VPSet(
        cms.PSet(
            Record = cms.string('SiStripApvGainRcd'),
            Label = cms.untracked.string(''),
            NormalizationFactor = cms.untracked.double(1.)
        ),
        cms.PSet(
            Record = cms.string('SiStripApvGain2Rcd'),
            Label = cms.untracked.string(''),
            NormalizationFactor = cms.untracked.double(1.)
        )
    )
)


