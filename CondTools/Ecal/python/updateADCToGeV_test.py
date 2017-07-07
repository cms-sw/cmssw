import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(100000000000),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(100000000000),
    interval = cms.uint64(1)
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:EcalADCToGeV.db'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDBCommon, 
  logconnect = cms.untracked.string('sqlite_file:log.db'),   
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalADCToGeVConstantRcd'),
      tag = cms.string('EcalADCToGeVConstant_V1_hlt')
    )
  )
)

process.Test1 = cms.EDAnalyzer("ExTestEcalADCToGeVAnalyzer",
  record = cms.string('EcalADCToGeVConstantRcd'),
  loggingOn= cms.untracked.bool(True),
  IsDestDbCheckedInQueryLog=cms.untracked.bool(True),
  SinceAppendMode=cms.bool(True),
  Source=cms.PSet(
    InputFile = cms.string('ADCtoGeV_Bon.xml'),
    firstRun = cms.string('98273'),
  )                            
)

process.p = cms.Path(process.Test1)
