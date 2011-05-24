import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_WEB'
#process.CondDBCommon.connect = 'sqlite_file:mytest.db'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
process.CondDBCommon.DBParameters.messageLevel = cms.untracked.int32(3)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(30),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(21),
    interval = cms.uint64(2)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('mytest')
    ))
)

process.mytest = cms.EDAnalyzer("IOVPayloadAnalyzer",
    record = cms.string('PedestalsRcd')
)

process.p = cms.Path(process.mytest)

