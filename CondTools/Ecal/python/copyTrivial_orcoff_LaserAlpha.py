import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("EcalTrivialAlpha_cfi")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_ECAL'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
process.CondDBCommon.connect = 'sqlite_file:LaserAlphasDB.db'

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
  timetype = cms.untracked.string('timestamp'),
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalLaserAlphasRcd'),
      tag = cms.string('EcalLaserAlphas_test_prompt')
    ), 
  )
)


#    timetype = cms.string('timestamp'),

process.dbCopy = cms.EDAnalyzer("EcalDBCopy",
  timetype = cms.string('timestamp'),
  toCopy = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalLaserAlphasRcd'),
      container = cms.string('EcalLaserAlphas')
    ), 
  )
)

process.prod = cms.EDAnalyzer("EcalTrivialObjectAnalyzer")

process.p = cms.Path(process.prod*process.dbCopy)
