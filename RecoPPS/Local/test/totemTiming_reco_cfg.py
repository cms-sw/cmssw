import FWCore.ParameterSet.Config as cms

process = cms.Process('CTPPS')

from RecoPPS.Local.PPSTimingCalibrationModeEnum_cff import PPSTimingCalibrationModeEnum
calibrationMode = PPSTimingCalibrationModeEnum.CondDB

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

if calibrationMode == PPSTimingCalibrationModeEnum.JSON:
    process.load('CondFormats.PPSObjects.ppsTimingCalibrationESSource_cfi')
    process.ppsTimingCalibrationESSource.calibrationFile = cms.FileInPath('RecoPPS/Local/data/timing_offsets_ufsd_2018.dec18.cal.json')
elif calibrationMode == PPSTimingCalibrationModeEnum.SQLite:
    # load calibrations from database
    process.load('CondCore.CondDB.CondDB_cfi')
    process.CondDB.connect = 'sqlite_file:totemTiming_calibration.sqlite' # SQLite input
    process.PoolDBESSource = cms.ESSource('PoolDBESSource',
        process.CondDB,
        DumpStats = cms.untracked.bool(True),
        toGet = cms.VPSet(
            cms.PSet(
                record = cms.string('PPSTimingCalibrationRcd'),
                tag = cms.string('TotemTimingCalibration')
            )
        )
    )

# raw data source
#process.source = cms.Source("NewEventStreamFileReader",
#    fileNames = cms.untracked.vstring(
#        '/store/t0streamer/Data/Physics/000/286/591/run286591_ls0521_streamPhysics_StorageManager.dat',
#        '/store/t0streamer/Minidaq/A/000/303/982/run303982_ls0001_streamA_StorageManager.dat',
#    )
#)
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning2018/ZeroBias/RAW/v1/000/314/816/00000/FCDB2DE6-4845-E811-91A1-FA163E6CD0D3.root',
    ),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# rechits production
process.load('Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi')
process.load('RecoPPS.Local.totemTimingLocalReconstruction_cff')

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("file:AOD.root"),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_totemTiming*_*_*',
    ),
)

# execution configuration
process.p = cms.Path(
    process.ctppsRawToDigi*
    process.totemTimingLocalReconstruction
)

process.outpath = cms.EndPath(process.output)
