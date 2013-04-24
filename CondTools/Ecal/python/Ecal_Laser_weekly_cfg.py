import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
#
# Choose the output database
#
#process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_311X_ECAL_LAS'
#process.CondDBCommon.connect = 'sqlite_file:DBLaser.db'
process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_ECAL'

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
  timetype = cms.untracked.string('runnumber'),
  toGet = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalLaserAPDPNRatiosRcd'),
      tag = cms.string('EcalLaserAPDPNRatios_weekly_v1_hlt')
    )
  )
 )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDBCommon,
  logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
  timetype = cms.untracked.string('runnumber'),
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalLaserAPDPNRatiosRcd'),
      tag = cms.string('EcalLaserAPDPNRatios_weekly_v1_hlt')
    )
  )
)
#
# Be sure to comment the following line while testing
#
#logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG'),

process.Test1 = cms.EDAnalyzer("ExTestEcalLaser_weekly_Analyzer",
  record = cms.string('EcalLaserAPDPNRatiosRcd'),
  Source = cms.PSet(
    debug = cms.bool(True),
  )
)

process.p = cms.Path(process.Test1)


