import FWCore.ParameterSet.Config as cms

process = cms.Process('test')

# minimum logging
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
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
process.CondDB.connect = 'sqlite_file:ppsDiamondTiming_calibrationLUT.sqlite' # SQLite input

process.PoolDBESSource = cms.ESSource('PoolDBESSource',
    process.CondDB,
    DumpStats = cms.untracked.bool(True),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('PPSTimingCalibrationLUTRcd'),
            tag = cms.string('PPSDiamondTimingCalibrationLUT')
        )
    )
)

process.ppsTimingCalibrationLUTAnalyzer = cms.EDAnalyzer('PPSTimingCalibrationLUTAnalyzer')

process.path = cms.Path(
    process.ppsTimingCalibrationLUTAnalyzer
)

