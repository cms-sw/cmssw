import FWCore.ParameterSet.Config as cms

##############################################################
#
# Laser corrections O2O. Transfer validated corrections, i.e.
# those obtained at most 40 hours before
#
##############################################################

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'
#
# Choose the output database
#
process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_42X_ECAL_LAS'
#process.CondDBCommon.connect = 'sqlite_file:DB.db'

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

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                      process.CondDBCommon,
                                      timetype = cms.untracked.string('timestamp'),
                                      toGet = cms.VPSet(cms.PSet(
    record = cms.string('EcalLaserAPDPNRatiosRcd'),
    tag = cms.string('EcalLaserAPDPNRatios_prompt')
    ))
                                      )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
                                          timetype = cms.untracked.string('timestamp'),
                                          toPut = cms.VPSet(cms.PSet(
    record = cms.string('EcalLaserAPDPNRatiosRcd'),
    tag = cms.string('EcalLaserAPDPNRatios_prompt')
    ))
                                          )
#
# Be sure to comment the following line while testing
#
process.PoolDBOutputService.logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG')

process.Test1 = cms.EDAnalyzer("ExTestEcalLaserAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('EcalLaserAPDPNRatiosRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
    # maxtime is mandatory
    # it can be expressed either as an absolute time with format YYYY-MM-DD HH24:MI:SS
    # or as a relative time w.r.t. now, using -N, where N is expressed in units
    # of hours
    maxtime = cms.string("-30"),
#       maxtime = cms.string("2011-12-12 23:59:59"),
        sequences = cms.string("20"),  
        OnlineDBUser = cms.string('CMS_ECAL_LASER_COND'),
    # debug must be False for production
        debug = cms.bool(False),
    # if fake is True, no insertion in the db is performed
        fake = cms.bool(False),
        OnlineDBPassword = cms.string('XXXXXXXXXXXXXX'),
        OnlineDBSID = cms.string('CMS_OMDS_LB')
    )
)

process.p = cms.Path(process.Test1)


