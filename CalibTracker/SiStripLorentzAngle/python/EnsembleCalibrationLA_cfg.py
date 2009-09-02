import FWCore.ParameterSet.Config as cms

process = cms.Process("MACRO")
process.add_(cms.Service("MessageLogger"))
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.calibration = cms.EDAnalyzer(
    "sistrip::EnsembleCalibrationLA",
    InputFiles = cms.vstring([]),
    InFileLocation = cms.string('/calibrationTree/tree'),
    Samples = cms.uint32(150),
    NBins = cms.uint32(50),
    LowBin = cms.double(-0.15),
    HighBin = cms.double(-0.02),
    Prefix = cms.untracked.string("peak_"),
    #MaxEvents = cms.untracked.uint32(4000),
    useWIDTH = cms.untracked.bool(True),
    useRATIO = cms.untracked.bool(True),
    useSQRTVAR = cms.untracked.bool(True),
    useSYMM = cms.untracked.bool(True)
    )

process.calibration.InputFiles += ["/d1/bbetchar/LA_calibration/ttbar_peak/calibTree_peak.root"
                                   "/d1/bbetchar/LA_calibration/zmumuj1_peak/res/calibTree_peak_*.root",
                                   "/d1/bbetchar/LA_calibration/zmumuj2_peak/res/calibTree_peak_*.root",
                                   "/d1/bbetchar/LA_calibration/zmumuj3_peak/res/calibTree_peak_*.root"
                                   ]

process.path = cms.Path(process.calibration)
