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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_hlt_relval', '')

if calibrationMode == PPSTimingCalibrationModeEnum.JSON:
    process.load('CondFormats.PPSObjects.ppsTimingCalibrationESSource_cfi')
    process.ppsTimingCalibrationESSource.calibrationFile = cms.FileInPath('RecoPPS/Local/data/timing_offsets_ufsd_2018.dec18.cal.json')
elif calibrationMode == PPSTimingCalibrationModeEnum.SQLite:
    # load calibrations from database
    process.load('CondCore.CondDB.CondDB_cfi')
    process.CondDB.connect = 'sqlite_file:ppsDiamondTiming_calibration.sqlite' # SQLite input
    process.PoolDBESSource = cms.ESSource('PoolDBESSource',
        process.CondDB,
        DumpStats = cms.untracked.bool(True),
        toGet = cms.VPSet(
            cms.PSet(
                record = cms.string('PPSTimingCalibrationRcd'),
                tag = cms.string('PPSDiamondTimingCalibration')
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
#        '/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/001C958C-7FA9-E711-858F-02163E011A5F.root',
#        '/store/data/Commissioning2018/ZeroBias/RAW/v1/000/314/816/00000/FCDB2DE6-4845-E811-91A1-FA163E6CD0D3.root',
        '/store/data/Run2018C/DoubleMuon/RAW/v1/000/319/525/00000/CE9C5CAC-AD85-E811-852B-FA163E26680F.root',
    ),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32( 1000 )

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoPPS.Configuration.recoCTPPS_cff")

# rechits production
process.load('Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi')
process.load('RecoPPS.Local.ctppsDiamondRecHits_cfi')

# local tracks fitter
process.load('RecoPPS.Local.ctppsDiamondLocalTracks_cfi')

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("file:AOD.root"),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_ctpps*_*_*',
    ),
)

# execution configuration
process.p = cms.Path(
    process.ctppsDiamondRawToDigi *
    process.ctppsDiamondLocalReconstruction
)

process.outpath = cms.EndPath(process.output)

