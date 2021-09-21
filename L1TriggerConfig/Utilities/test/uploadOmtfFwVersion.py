import FWCore.ParameterSet.Config as cms

process = cms.Process("tester")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("EmptySource", firstRun = cms.untracked.uint32(3))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("L1Trigger.L1TMuonOverlap.fakeOmtfFwVersion_cff")

process.getter = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(
       cms.PSet(
           record = cms.string('L1TMuonOverlapFwVersionRcd'),
           data   = cms.vstring('L1TMuonOverlapFwVersion')
       )
   ),
   verbose = cms.untracked.bool(True)
)

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string('sqlite:l1config.db')

outputDB = cms.Service("PoolDBOutputService",
                       CondDB,
                       toPut   = cms.VPSet(
                           cms.PSet(
                               record = cms.string('L1TMuonOverlapFwVersionRcd'),
                               tag = cms.string('L1TMuonOverlapFwVersionPrototype_Stage2v0_hlt')
                           )
                       )
)
outputDB.DBParameters.authenticationPath = '.'
process.add_(outputDB)

process.l1opw = cms.EDAnalyzer("L1TMuonOverlapFwVersionTester", isO2Opayload = cms.untracked.bool(False), writeToDB = cms.untracked.bool(True))

process.p = cms.Path(process.getter + process.l1opw)

