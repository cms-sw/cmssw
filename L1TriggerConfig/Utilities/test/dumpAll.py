import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load('L1Trigger.L1TMuonEndCap.fakeEmtfParams_cff')
process.load('L1Trigger.L1TMuonBarrel.fakeBmtfParams_cff')
process.load('L1Trigger.L1TMuon.fakeGmtParams_cff')
#process.load('L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_3_HI_cfi')
process.load('L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_3_cfi')

# 2016_3_3 and 3_3_HI

process.l1ew = cms.EDAnalyzer("L1TMuonEndCapParamsWriter") ## What does this string match / refer to? - AWB 10.04.18
process.l1bw = cms.EDAnalyzer("L1TMuonBarrelParamsWriter")
process.l1gw = cms.EDAnalyzer("L1TMuonGlobalParamsWriter")
process.l1cw = cms.EDAnalyzer("L1TCaloStage2ParamsWriter")

from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
outputDB = cms.Service("PoolDBOutputService",
                       CondDBSetup,
                       connect = cms.string('sqlite_file:l1configPP.db'),
                       toPut   = cms.VPSet(
                           cms.PSet(
                               record = cms.string('L1TMuonEndCapParamsRcd'),
                               tag = cms.string("L1TMuonEndCapParams_static_v91.10")
                           ),
                           cms.PSet(
                               record = cms.string('L1TMuonBarrelParamsRcd'),
                               tag = cms.string("L1TMuonBarrelParams_static_v91.10")
                           ),
                           cms.PSet(
                               record = cms.string('L1TMuonGlobalParamsRcd'),
                               tag = cms.string("L1TMuonGlobalParams_static_v91.10")
                           ),
                           cms.PSet(
                               record = cms.string('L1TCaloStage2ParamsRcd'),
                               tag = cms.string("L1TCaloStage2Params_static_v91.10_v3_3")
                           )
                       )
)
outputDB.DBParameters.authenticationPath = '.'
process.add_(outputDB)

process.p = cms.Path(process.l1ew + process.l1bw + process.l1gw + process.l1cw)
