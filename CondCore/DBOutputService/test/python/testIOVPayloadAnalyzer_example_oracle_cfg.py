import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'oracle://cms_orcoff_prep/CMS_COND_WEB'
process.CondDB.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
#process.CondDB.DBParameters.messageLevel = cms.untracked.int32(3)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(30),
    timetype = cms.string('Run'),
    firstValue = cms.uint64(21),
    interval = cms.uint64(2)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('Run'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('mytest')
    ))
)

process.mytest = cms.EDAnalyzer("IOVPayloadAnalyzer",
    record = cms.string('PedestalsRcd')
)

process.p = cms.Path(process.mytest)

