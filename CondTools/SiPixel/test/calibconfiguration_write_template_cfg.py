import FWCore.ParameterSet.Config as cms

process = cms.Process("PixelCalibTemplate")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(PIXELRUNNUMBER),
    lastValue = cms.uint64(PIXELRUNNUMBER),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelCalibConfigurationRcd'),
        tag = cms.string('PIXELTAG')
    ))
)

process.SiPixelCalibConfigurationObjectMaker = cms.EDFilter("SiPixelCalibConfigurationObjectMaker",
    inputFileName = cms.untracked.string('PIXELFILENAME')
)

process.p1 = cms.Path(process.SiPixelCalibConfigurationObjectMaker)
process.CondDBCommon.connect = 'sqlite_file:siPixelCalibConfiguration.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'
process.CondDBCommon.DBParameters.messageLevel = 0


