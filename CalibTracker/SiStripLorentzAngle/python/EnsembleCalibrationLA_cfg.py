import FWCore.ParameterSet.Config as cms

process = cms.Process("MACRO")
process.add_(cms.Service("MessageLogger"))
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

from CalibTracker.SiStripLorentzAngle.MeasureLA_cff import METHOD_MULTI, METHOD_SYMM, METHOD_WIDTH, METHOD_RATIO, METHOD_SQRTVAR

process.calibration = cms.EDAnalyzer(
    "sistrip::EnsembleCalibrationLA",
    InputFiles = cms.vstring([]),
    InFileLocation = cms.string('/calibrationTree/tree'),
    Samples = cms.uint32(100),
    NBins = cms.uint32(14),
    LowBin = cms.double(-0.15),
    HighBin = cms.double(-0.0),
    Prefix = cms.untracked.string("peak_"),
    #MaxEvents = cms.untracked.uint32(9000),
    Methods = cms.vint32(METHOD_MULTI, METHOD_SYMM, METHOD_WIDTH, METHOD_RATIO, METHOD_SQRTVAR)
    )

process.calibration.InputFiles += ["/d2/bbetchar/LA_calibration/ttbar_peak/calibTree_peak.root"
                                   "/d2/bbetchar/LA_calibration/zmumuj1_peak/res/calibTree_peak_*.root",
                                   "/d2/bbetchar/LA_calibration/zmumuj2_peak/res/calibTree_peak_*.root",
                                   "/d2/bbetchar/LA_calibration/zmumuj3_peak/res/calibTree_peak_*.root"
                                   ]

process.path = cms.Path(process.calibration)
