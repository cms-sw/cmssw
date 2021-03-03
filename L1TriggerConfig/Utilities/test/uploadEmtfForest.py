import FWCore.ParameterSet.Config as cms

process = cms.Process("tester")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("EmptySource", firstRun = cms.untracked.uint32(3))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("L1Trigger.L1TMuonEndCap.fakeEmtfParams_cff")

process.getter = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(
       cms.PSet(
           record = cms.string('L1TMuonEndCapForestRcd'),
           data   = cms.vstring('L1TMuonEndCapForest')
       )
   ),
   verbose = cms.untracked.bool(True)
)

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string('sqlite:l1config.db')
#CondDB.connect = cms.string('oracle://cms_orcoff_prep/CMS_CONDITIONS')

outputDB = cms.Service("PoolDBOutputService",
                       CondDB,
                       toPut   = cms.VPSet(
                           cms.PSet(
                               record = cms.string('L1TMuonEndCapForestRcd'),
                               tag = cms.string('L1TMuonEndCapForest')
                           )
                       )
)
outputDB.DBParameters.authenticationPath = '.'
process.add_(outputDB)

process.l1bpw = cms.EDAnalyzer("L1TMuonEndCapForestWriter", isO2Opayload = cms.untracked.bool(False))

process.p = cms.Path(process.getter + process.l1bpw)

