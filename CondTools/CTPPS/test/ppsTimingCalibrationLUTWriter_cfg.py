import FWCore.ParameterSet.Config as cms

process = cms.Process('test')

process.source = cms.Source('EmptyIOVSource',
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)


# load calibrations from JSON file
process.load('CalibPPS.ESProducers.ppsTimingCalibrationLUTESSource_cfi')
process.ppsTimingCalibrationLUTESSource.calibrationFile = cms.FileInPath('CalibPPS/TimingCalibration/data/LUT_test.json')


# output service for database
process.load('CondCore.CondDB.CondDB_cfi')
process.CondDB.connect = 'sqlite_file:ppsDiamondTiming_calibrationLUT.sqlite' # SQLite output

process.PoolDBOutputService = cms.Service('PoolDBOutputService',
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('PPSTimingCalibrationLUTRcd'),
            tag = cms.string('PPSDiamondTimingCalibrationLUT'),
        )
    )
)

process.ppsTimingCalibrationLUTWriter = cms.EDAnalyzer('PPSTimingCalibrationLUTWriter')

process.path = cms.Path(
    process.ppsTimingCalibrationLUTWriter
)

