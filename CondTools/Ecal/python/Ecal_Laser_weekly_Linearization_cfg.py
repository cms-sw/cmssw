import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
#
# Choose the output database
#
process.CondDB.connect = 'sqlite_file:EcalLin.db'
#process.CondDB.connect = 'oracle://cms_orcoff_prep/CMS_CONDITIONS'

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
      record = cms.string('EcalTPGLinearizationConstRcd'),
      tag = cms.string('EcalTPGLinearizationConst_weekly_hlt')
    )
  )
 )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDB,
  logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
  timetype = cms.untracked.string('runnumber'),
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalTPGLinearizationConstRcd'),
      tag = cms.string('EcalTPGLinearizationConst_weekly_hlt')
    )
  )
)

process.Test1 = cms.EDAnalyzer("ExTestEcalLaser_weekly_Linearization_Analyzer",
  record = cms.string('EcalTPGLinearizationConstRcd'),
  Source = cms.PSet(
    debug = cms.bool(True),
  )
)

process.p = cms.Path(process.Test1)


