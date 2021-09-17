import FWCore.ParameterSet.Config as cms

process = cms.Process('test')

process.source = cms.Source('EmptyIOVSource',
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

from CondFormats.PPSObjects.PPSTimingDetEnum_cff import PPSTimingDetEnum

# load calibrations from JSON file
process.load('CalibPPS.ESProducers.ppsTimingCalibrationESSource_cfi')
process.ppsTimingCalibrationESSource.calibrationFile = cms.FileInPath('RecoPPS/Local/data/timing_calibration_diamond_2018_mar19.ex.json')
process.ppsTimingCalibrationESSource.subDetector = PPSTimingDetEnum.PPS_DIAMOND

# output service for database
process.load('CondCore.CondDB.CondDB_cfi')
process.CondDB.connect = 'sqlite_file:ppsDiamondTiming_calibration.sqlite' # SQLite output

process.PoolDBOutputService = cms.Service('PoolDBOutputService',
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('PPSTimingCalibrationRcd'),
            tag = cms.string('PPSDiamondTimingCalibration'),
        )
    )
)

process.ppsTimingCalibrationWriter = cms.EDAnalyzer('PPSTimingCalibrationWriter')

process.path = cms.Path(
    process.ppsTimingCalibrationWriter
)

