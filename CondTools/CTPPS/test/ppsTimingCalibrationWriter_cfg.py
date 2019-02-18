import FWCore.ParameterSet.Config as cms

process = cms.Process('test')

process.source = cms.Source('EmptyIOVSource',
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

# load calibrations from JSON file
process.load('CondFormats.CTPPSReadoutObjects.ppsTimingCalibrationESSource_cfi')
process.ppsTimingCalibrationESSource.calibrationFile = cms.FileInPath('RecoCTPPS/TotemRPLocal/data/timing_offsets_ufsd_2018.dec18.cal.json')

# output service for database
process.load('CondCore.CondDB.CondDB_cfi')
process.CondDB.connect = 'sqlite_file:totemTiming_calibration.sqlite' # SQLite output
#process.CondDB.connect = 'oracle://cmsprep/TimingCalibration' # Oracle output
#process.CondDB.connect = 'frontier://cmsfrontier.cern.ch:8000/FrontierPrep/TimingCalibration' # Frontier output

process.PoolDBOutputService = cms.Service('PoolDBOutputService',
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('PPSTimingCalibrationRcd'),
            tag = cms.string('TotemTimingCalibration'),
        )
    )
)

process.ppsTimingCalibrationWriter = cms.EDAnalyzer('PPSTimingCalibrationWriter')

process.path = cms.Path(
    process.ppsTimingCalibrationWriter
)

