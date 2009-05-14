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

#    logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_21X_POPCONLOG'),



process.Test1 = cms.EDAnalyzer("ExTestEcalLaserAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('EcalLaserAPDPNRatiosRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
        GenTag = cms.string('LOCAL'),
        firstRun = cms.string('65934'),
        lastRun = cms.string('65936'),
        LocationSource = cms.string('P5'),
        OnlineDBUser = cms.string('CMS_ECAL_R'),
        debug = cms.bool(True),
        Location = cms.string('P5_Co'),
        OnlineDBPassword = cms.string('********'),
        OnlineDBSID = cms.string('CMSR')
    )
)

process.p = cms.Path(process.Test1)


