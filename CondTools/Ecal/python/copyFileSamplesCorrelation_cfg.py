import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("EcalSamplesCorrelation_cfi")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:EcalSamplesCorrelationFile.db'

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
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalSamplesCorrelationRcd'),
      tag = cms.string('EcalSamplesCorrelation_mc')
#      tag = cms.string('EcalSamplesCorrelationFile')
    )
  )
)

process.dbCopy = cms.EDAnalyzer("EcalDBCopy",
  timetype = cms.string('runnumber'),
  toCopy = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalSamplesCorrelationRcd'),
      container = cms.string('EcalSamplesCorrelation')
    )
  )
)

process.prod = cms.EDAnalyzer("EcalTrivialObjectAnalyzer")

process.p = cms.Path(process.prod*process.dbCopy)
