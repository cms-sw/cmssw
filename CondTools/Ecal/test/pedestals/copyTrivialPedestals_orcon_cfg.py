import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("EcalTrivialCondRetriever_cfi")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
# process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_ECAL'
process.CondDBCommon.connect = cms.string('oracle://cms_orcon_prod/CMS_COND_31X_ECAL')
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/nfshome0/xiezhen/conddb')
# process.CondDBCommon.connect = 'sqlite_file:DB.db'
# process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'


process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        tag = cms.string('EcalPedestals_v01_mc')
        ))
)

process.dbCopy = cms.EDAnalyzer("EcalDBCopy",
    timetype = cms.string('runnumber'),
    toCopy = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        container = cms.string('EcalPedestals')
        ))
)



process.p = cms.Path(process.dbCopy)

