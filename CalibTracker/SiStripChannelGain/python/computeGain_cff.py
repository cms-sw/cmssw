import FWCore.ParameterSet.Config as cms

SiStripCalib = cms.EDFilter("SiStripGainFromCalibTree",
    OutputGains         = cms.string('Gains_ASCII.txt'),

    minTrackMomentum    = cms.untracked.double(2),
    minNrEntries        = cms.untracked.double(25),
    maxChi2OverNDF      = cms.untracked.double(9999999.0),
    maxMPVError         = cms.untracked.double(25.0),
    maxNrStrips         = cms.untracked.uint32(8),

    Validation          = cms.untracked.bool(False),
    OldGainRemoving     = cms.untracked.bool(False),
    FirstSetOfConstants = cms.untracked.bool(True),

    CalibrationLevel    = cms.untracked.int32(0), # 0==APV, 1==Laser, 2==module

    InputFiles          = cms.vstring(
#	"rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_20_134721_1.root",
        "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_20_134721_1.root",
        "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_21_134721_1.root",
        "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_22_134721_1.root",
        "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_23_134721_1.root",
        "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_24_134721_1.root",
        "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_25_134721_1.root",
        "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_26_134721_1.root",
        "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_27_134721_1.root",
        "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_28_134721_1.root",
        "rfio:/castor/cern.ch/cms/store/group/tracker/strip/calibration/calibrationtree/calibTree_29_134721_1.root",
    ),

    UseCalibration     = cms.untracked.bool(False),
    calibrationPath    = cms.untracked.string(""),

    SinceAppendMode     = cms.bool(True),
    IOVMode             = cms.string('Job'),
    Record              = cms.string('SiStripApvGainRcd'),
    doStoreOnDB         = cms.bool(True)
)

SiStripCalibValidation = SiStripCalib.clone()
SiStripCalibValidation.OutputGains         = cms.string('Validation_ASCII.txt') 
SiStripCalibValidation.UseCalibration      = cms.untracked.bool(True)
SiStripCalibValidation.calibrationPath     = cms.untracked.string("file:Gains.root") 
SiStripCalibValidation.doStoreOnDB         = cms.bool(False) 
