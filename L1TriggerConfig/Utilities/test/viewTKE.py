import FWCore.ParameterSet.Config as cms

process = cms.Process("tester")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("EmptySource", firstRun = cms.untracked.uint32(3))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
process.l1conddb = cms.ESSource("PoolDBESSource",
       CondDBSetup,
       connect = cms.string('sqlite:o2o/l1config.db'),
       toGet   = cms.VPSet(
            cms.PSet(
                 record = cms.string('L1TriggerKeyExtRcd'),
                 tag = cms.string("L1TriggerKeyExt_Stage2v0_hlt")
            )
       )
)

process.l1cr = cms.EDAnalyzer("L1TriggerKeyExtReader", label = cms.string("") )

process.p = cms.Path(process.l1cr)

