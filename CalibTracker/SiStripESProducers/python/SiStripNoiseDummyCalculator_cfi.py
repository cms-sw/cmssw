import FWCore.ParameterSet.Config as cms

siStripNoiseDummyCalculator = cms.EDFilter("SiStripNoiseDummyCalculator",
    #relevant if striplenght mode is chosen
    # standard value for deconvolution mode is 51. For peak mode 38.8.
    NoiseStripLengthSlope = cms.double(51.0),
    badStripProbability = cms.double(0.0),
    MeanNoise = cms.double(4.0),
    printDebug = cms.untracked.bool(False),
    # standard value for deconvolution mode is 630. For peak mode  414.
    NoiseStripLengthQuote = cms.double(630.0),
    StripLengthMode = cms.bool(True),
    electronPerAdc = cms.double(250.0),
    IOVMode = cms.string('Run'),
    Record = cms.string('SiStripNoisesRcd'),
    doStoreOnDB = cms.bool(True),
    SigmaNoise = cms.double(0.5),
    #relevant if random mode is chosen
    MinPositiveNoise = cms.double(0.1),
    #cards relevant to mother class
    SinceAppendMode = cms.bool(True)
)


