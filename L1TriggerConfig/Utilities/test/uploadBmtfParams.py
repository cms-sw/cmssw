import FWCore.ParameterSet.Config as cms

process = cms.Process("tester")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("EmptySource", firstRun = cms.untracked.uint32(10))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

## these lines load ESProducers to be run outside the path (automatically)
process.load("L1Trigger.L1TMuonBarrel.fakeBmtfParams_cff")
process.load("L1Trigger.L1TMuonBarrel.staticKBmtfParams_cff")

##this is general code for getting rcds and datas loaded to the ESSetup
process.getter = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(
       cms.PSet(
           record = cms.string('L1TMuonBarrelParamsRcd'),
           data   = cms.vstring('L1TMuonBarrelParams')
       ),
       cms.PSet(
           record = cms.string('L1TMuonBarrelKalmanParamsRcd'),
           data = cms.vstring('L1TMuonBarrelKalmanParams')
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
                               record = cms.string('L1TMuonBarrelParamsRcd'),
                               tag = cms.string('L1TMuonBarrelParamsPrototype_Stage2v0_hlt')
                           ),
                           cms.PSet(
                               record = cms.string('L1TMuonBarrelKalmanParamsRcd'),
                               tag = cms.string('L1TMuonBarrelKalmanParamsPrototype_Stage2v0_hlt')
                           )
                       )
)
outputDB.DBParameters.authenticationPath = '.'
process.add_(outputDB)

process.l1bpw = cms.EDAnalyzer("L1TMuonBarrelParamsWriter",
                               isO2Opayload = cms.untracked.bool(False))

process.p = cms.Path(process.getter + process.l1bpw)
