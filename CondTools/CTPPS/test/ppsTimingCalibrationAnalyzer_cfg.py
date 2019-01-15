import FWCore.ParameterSet.Config as cms

process = cms.Process('test')

# minimum of logs
process.MessageLogger = cms.Service('MessageLogger',
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    )
)

process.source = cms.Source('EmptyIOVSource',
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

# load calibrations from database
process.load('CondCore.CondDB.CondDB_cfi')
# SQLite input
process.CondDB.connect = 'sqlite_file:totemTiming_calibration.sqlite'

process.PoolDBESSource = cms.ESSource('PoolDBESSource',
    process.CondDB,
    DumpStats = cms.untracked.bool(True),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('PPSTimingCalibrationRcd'),
            tag = cms.string('TotemTimingCalibration'),
            #label = cms.string('UFSD')
        )
    )
)

process.ppsTimingCalibrationAnalyzer = cms.EDAnalyzer('PPSTimingCalibrationAnalyzer')

process.path = cms.Path(
    process.ppsTimingCalibrationAnalyzer
)

