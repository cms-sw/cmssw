import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_WEB')
process.CondDB.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/test'

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('Run'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    timetype = cms.untracked.string('Run'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('ped_tag')
    ), cms.PSet(
        record = cms.string('mySiStripNoisesRcd'),
        tag = cms.string('noise_tag')
    ))
)

process.mytest = cms.EDAnalyzer("writeMultipleRecords",
                                PedCallerName=cms.string('PedestalsRcd'),
                                StripCallerName=cms.string('mySiStripNoisesRcd')
)

process.p = cms.Path(process.mytest)


