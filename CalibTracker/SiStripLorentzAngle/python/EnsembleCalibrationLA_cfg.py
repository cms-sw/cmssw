import FWCore.ParameterSet.Config as cms

process = cms.Process("MACRO")
process.add_(cms.Service("MessageLogger"))
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

from CalibTracker.SiStripLorentzAngle.MeasureLA_cff import METHOD_WIDTH, METHOD_PROB1, METHOD_AVGV2, METHOD_AVGV3, METHOD_RMSV2, METHOD_RMSV3

process.load('Configuration.Geometry.GeometryIdeal_cff')

process.calibration = cms.EDAnalyzer(
    "sistrip::EnsembleCalibrationLA",
    InputFiles = cms.vstring([]),
    InFileLocation = cms.string('/calibrationTree/tree'),
    Samples = cms.uint32(10),
    NBins = cms.uint32(5),
    LowBin = cms.double(-0.16),
    HighBin = cms.double(-0.0),
    Prefix = cms.untracked.string("peak_"),
    #MaxEvents = cms.untracked.uint32(9000),
    Methods = cms.vint32(METHOD_WIDTH.value(), METHOD_PROB1.value(), METHOD_AVGV2.value(), METHOD_AVGV3.value(), METHOD_RMSV2.value(), METHOD_RMSV3.value())
    )

process.calibration.InputFiles += ["/d1/bbetchar/LorentzAngle/calibration/ttbar_peak/calibTree_peak.root"]

process.path = cms.Path(process.calibration)
