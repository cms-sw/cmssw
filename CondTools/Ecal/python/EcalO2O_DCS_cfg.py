import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
# process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_ECAL'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
process.CondDBCommon.connect = 'sqlite_file:DB.db'



process.MessageLogger = cms.Service("MessageLogger",
                                        debugModules = cms.untracked.vstring('*'),
                                        destinations = cms.untracked.vstring('cout')
                                    )

process.source = cms.Source("EmptyIOVSource",
                                firstValue = cms.uint64(1000000),
                                lastValue = cms.uint64(1000000),
                                timetype = cms.string('runnumber'),
                                interval = cms.uint64(1)
                            )

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalDCSTowerStatusRcd'),
        tag = cms.string('EcalDCSTowerStatus_mc')
    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalDCSTowerStatusRcd'),
        tag = cms.string('EcalDCSTowerStatus_mc')
    ))
)

#    logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_21X_POPCONLOG'),



process.Test1 = cms.EDAnalyzer("ExTestEcalDCSAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('EcalDCSTowerStatusRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
        firstRun = cms.string('42058'),
        lastRun = cms.string('100000000'),
        OnlineDBUser = cms.string('******'),
                debug = cms.bool(True),
                OnlineDBPassword = cms.string('*******'),
                OnlineDBSID = cms.string('*******')
            )
    )


process.p = cms.Path(process.Test1)


