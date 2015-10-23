import FWCore.ParameterSet.Config as cms

SiStripCalib = cms.EDAnalyzer("SiStripGainFromCalibTree",
    OutputGains         = cms.string('Gains_ASCII.txt'),
    Tracks              = cms.untracked.InputTag('CalibrationTracksRefit'),
    AlgoMode            = cms.untracked.string('CalibTree'),

    minTrackMomentum    = cms.untracked.double(2),
    minNrEntries        = cms.untracked.double(25),
    maxChi2OverNDF      = cms.untracked.double(9999999.0),
    maxMPVError         = cms.untracked.double(25.0),
    maxNrStrips         = cms.untracked.uint32(8),

    Validation          = cms.untracked.bool(False),
    OldGainRemoving     = cms.untracked.bool(False),
    FirstSetOfConstants = cms.untracked.bool(True),

    CalibrationLevel    = cms.untracked.int32(0), # 0==APV, 1==Laser, 2==module

    InputFiles          = cms.untracked.vstring(),

    UseCalibration     = cms.untracked.bool(False),
    calibrationPath    = cms.untracked.string(""),

    saveSummary         = cms.untracked.bool(False),

    GoodFracForTagProd  = cms.untracked.double(0.98),
    NClustersForTagProd = cms.untracked.double(1E8),
    

    SinceAppendMode     = cms.bool(True),
    TimeFromEndRun      = cms.untracked.bool(True),
    IOVMode             = cms.string('Job'),
    Record              = cms.string('SiStripApvGainRcd'),
    doStoreOnDB         = cms.bool(True)
)

SiStripCalibValidation = SiStripCalib.clone()
SiStripCalibValidation.OutputGains         = cms.string('Validation_ASCII.txt') 
SiStripCalibValidation.UseCalibration      = cms.untracked.bool(True)
SiStripCalibValidation.calibrationPath     = cms.untracked.string("file:Gains.root") 
SiStripCalibValidation.doStoreOnDB         = cms.bool(False) 
