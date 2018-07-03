import FWCore.ParameterSet.Config as cms

process = cms.Process("MACRO")
process.add_(cms.Service("MessageLogger"))
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("CalibTracker.SiStripLorentzAngle.MeasureLA_cfi")
process.MeasureLA.TrackerParameters = cms.FileInPath("Geometry/TrackerCommonData/data/trackerParameters.xml")
process.MeasureLA.InputFiles += ["/d2/bbetchar/LA_calibration/craft09ALCA/res/calibTree_peak_*.root"]
process.MeasureLA.MaxEvents = cms.untracked.uint32(100000)
