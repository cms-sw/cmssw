import FWCore.ParameterSet.Config as cms

process = cms.Process("Calibration")

process.load("CalibMuon.DTCalibration.messageLoggerDebug_cff")
process.MessageLogger.debugModules = cms.untracked.vstring('dtVDriftSegmentCalibration')

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load("CalibMuon.DTCalibration.dt_offlineAnalysis_common_vDriftSeg_cff")

process.load("CalibMuon.DTCalibration.dtVDriftSegmentCalibration_cfi")

"""
process.p = cms.Path(process.muonDTDigis*
                     process.dt1DRecHits*process.dt2DSegments*process.dt4DSegments*
                     process.dtVDriftSegmentCalibration)
"""
process.p = cms.Path(process.dt1DRecHits*process.dt2DSegments*process.dt4DSegments*
                     process.dtVDriftSegmentCalibration)
