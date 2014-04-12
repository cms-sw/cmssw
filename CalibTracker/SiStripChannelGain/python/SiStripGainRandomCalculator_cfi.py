import FWCore.ParameterSet.Config as cms

siStripGainRandomCalculator = cms.EDFilter("SiStripGainRandomCalculator",
    MeanGain = cms.double(1.0),
    printDebug = cms.untracked.bool(False),
    IOVMode = cms.string('Run'),
    Record = cms.string('SiStripApvGainRcd'),
    doStoreOnDB = cms.bool(True),
    SigmaGain = cms.double(0.0),
    MinPositiveGain = cms.double(0.1),
    SinceAppendMode = cms.bool(True)
)


