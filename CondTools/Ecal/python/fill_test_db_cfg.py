import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_ECAL'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
process.CondDBCommon.connect = 'sqlite_file:DB.db'

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    ),
    debugModules = cms.untracked.vstring('*')
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
        tag = cms.string('EcalPedestals_test')
    ))
)

process.dbCopy = cms.EDAnalyzer("EcalTestDevDB",
    lastRun = cms.string('5'),
    timetype = cms.string('runnumber'),
    firstRun = cms.string('1'),
    toCopy = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        container = cms.string('EcalPedestals')
    )),
    interval = cms.string('1')
)

process.p = cms.Path(process.dbCopy)


