import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
#
# Choose the output database
#
process.CondDBCommon.connect = 'sqlite_file:EcalTPGFineGrainStrip.db'

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
      record = cms.string('EcalTPGFineGrainStripEERcd'),
      tag = cms.string('EcalTPGFineGrainStrip_test')
    )
  )
 )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDBCommon,
  logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
  timetype = cms.untracked.string('runnumber'),
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalTPGFineGrainStripEERcd'),
      tag = cms.string('EcalTPGFineGrainStrip_test')
    )
  )
)

process.Test1 = cms.EDAnalyzer("ExTestEcalTPGFineGrainStripfromFile",
  record = cms.string('EcalTPGFineGrainStripEERcd'),
  Source = cms.PSet(
    debug = cms.bool(True),
  )
)

process.p = cms.Path(process.Test1)
