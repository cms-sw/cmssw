import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_311X_ECAL_LAS'
process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'
#process.CondDBCommon.connect = 'sqlite_file:DB.db'



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

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    timetype = cms.untracked.string('timestamp'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalLaserAPDPNRatiosRcd'),
        tag = cms.string('EcalLaserAPDPNRatios_mc')
    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
    timetype = cms.untracked.string('timestamp'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalLaserAPDPNRatiosRcd'),
        tag = cms.string('EcalLaserAPDPNRatios_mc')
    ))
)

logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG'),



process.Test1 = cms.EDAnalyzer("ExTestEcalLaserAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('EcalLaserAPDPNRatiosRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
        maxtime = cms.string("2011-12-31 23:59:59"),
        sequences = cms.string("16"),  
        OnlineDBUser = cms.string('CMS_ECAL_LASER_COND'),
        debug = cms.bool(True),
        OnlineDBPassword = cms.string('XXX'),
        OnlineDBSID = cms.string('CMS_OMDS_LB')
    )
)

process.p = cms.Path(process.Test1)


