import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_31X_ECAL'
#process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'
process.CondDB.connect = 'sqlite_file:DB.db'

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
  process.CondDB,
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalClusterEnergyCorrectionParametersRcd'),
      tag = cms.string('EcalClusterEnergyCorrectionParameters_2011_mc')
    )
  )
)

process.dbCopy = cms.EDAnalyzer("EcalDBCopy",
  timetype = cms.string('runnumber'),
  toCopy = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalClusterEnergyCorrectionParametersRcd'),
      container = cms.string('EcalClusterEnergyCorrectionParameters')
    )
  )
)

process.prod = cms.EDAnalyzer("EcalTrivialObjectAnalyzer")

process.p = cms.Path(process.prod*process.dbCopy)
