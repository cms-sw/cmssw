import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelCalibConfTest")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:siPixelCalibConfiguration.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'
process.CondDBCommon.DBParameters.messageLevel = 0

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    )
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelCalibConfigurationRcd'),
        tag = cms.string('SiPixelCalibConfiguration_test')
    ))
)

process.SiPixelCalibConfigurationObjectMaker = cms.EDAnalyzer("SiPixelCalibConfigurationObjectMaker",
    inputFileName = cms.untracked.string('/afs/cern.ch/cms/Tracker/Pixel/forward/FPIX/HC+Z1/calib_PixelAlive_281.dat')
)

process.p1 = cms.Path(process.SiPixelCalibConfigurationObjectMaker)



