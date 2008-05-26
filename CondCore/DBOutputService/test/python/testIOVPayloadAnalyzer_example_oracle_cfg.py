import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'oracle://cms_orcoff_int2r/CMS_COND_PRESH'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(10),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
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

