import FWCore.ParameterSet.Config as cms

process = cms.Process("PixelCalibTemplate")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'oracle://cms_orcoff_prep/CMS_COND_PIXEL_COMM_21X'
process.CondDB.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
process.CondDB.DBParameters.messageLevel = 0


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
    process.CondDB,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelCalibConfigurationRcd'),
        tag = cms.string('PIXELTAG')
    ))
)

process.SiPixelCalibConfigurationObjectMaker = cms.EDAnalyzer("SiPixelCalibConfigurationObjectMaker",
    inputFileName = cms.untracked.string('PIXELFILENAME')
)

process.p1 = cms.Path(process.SiPixelCalibConfigurationObjectMaker)
