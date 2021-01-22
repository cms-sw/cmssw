#copied from git cmssw/CondTools/Ecal/python/updateIntercali.py
import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('*')
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(100000000000),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(100000000000),
    interval = cms.uint64(1)
)

process.load("CondCore.CondDB.CondDB_cfi")

process.CondDB.connect = 'sqlite_file:EcalPFRecHitThresholds_34sigma_TL235.db'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDB, 
  logconnect = cms.untracked.string('sqlite_file:log.db'),   
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalPFRecHitThresholdsRcd'),
      tag = cms.string('EcalPFRecHitThresholds_34sigma_TL235')
    )
  )
)

process.Test1 = cms.EDAnalyzer("ExTestEcalPFRecHitThresholdsAnalyzer",
  record = cms.string('EcalPFRecHitThresholdsRcd'),
  loggingOn= cms.untracked.bool(True),
  IsDestDbCheckedInQueryLog=cms.untracked.bool(True),
  SinceAppendMode=cms.bool(True),
  Source=cms.PSet(
    firstRun = cms.string('1'),
    type = cms.string('txt'),
    fileName = cms.string('product_TL235.txt'),
#    type = cms.string('xml'),
#    fileName = cms.string('Intercalib_Bon.xml'),
  )                            
)

process.p = cms.Path(process.Test1)
