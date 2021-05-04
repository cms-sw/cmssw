import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
#
# Choose the output database
#
#process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_CONDITIONS'
process.CondDB.connect = 'sqlite_file:DBLaser.db'

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
  process.CondDB,
  timetype = cms.untracked.string('runnumber'),
  toGet = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalLaserAPDPNRatiosRcd'),
      tag = cms.string('EcalLaserAPDPNRatios_weekly_v1_hlt')
    )
  )
 )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDB,
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


