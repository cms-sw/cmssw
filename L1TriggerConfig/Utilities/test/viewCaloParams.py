import FWCore.ParameterSet.Config as cms

process = cms.Process("tester")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("EmptySource", firstRun = cms.untracked.uint32(3))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

#process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2016_v2_2_cfi")

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string('sqlite:l1config.db')

process.l1conddb = cms.ESSource("PoolDBESSource",
       CondDB,
       toGet   = cms.VPSet(
            cms.PSet(
                 record = cms.string('L1TCaloParamsRcd'),
                 tag = cms.string("L1TCaloParams_Stage2v0_hlt")
            )
       )
)

process.l1cpv = cms.EDAnalyzer("L1TCaloParamsViewer", printEgIsoLUT = cms.untracked.bool(False) )

process.p = cms.Path(process.l1cpv)

