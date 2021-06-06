import FWCore.ParameterSet.Config as cms

process = cms.Process("tester")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("EmptySource", firstRun = cms.untracked.uint32(284074)) #268315 271658
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
process.l1conddb = cms.ESSource("PoolDBESSource",
       CondDBSetup,
       connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
       toGet   = cms.VPSet(
            cms.PSet(
                 record = cms.string('L1TriggerKeyExtRcd'),
                 tag = cms.string("L1TriggerKeyExt_Stage2v0_hlt")
            ),
            cms.PSet(
                 record = cms.string('L1TriggerKeyListExtRcd'),
                 tag = cms.string("L1TriggerKeyListExt_Stage2v0_hlt")
            ),
            cms.PSet(
                 record  = cms.string("L1TUtmTriggerMenuRcd"),
                 tag     = cms.string("L1TUtmTriggerMenu_Stage2v0_hlt"),
            )
       )
)

#process.load("CondCore.CondDB.CondDB_cfi")
from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string('sqlite:l1config.db')

outputDB = cms.Service("PoolDBOutputService",
    CondDB,
    toPut   = cms.VPSet(
        cms.PSet(
            record = cms.string('L1TriggerKeyExtRcd'),
            tag = cms.string('L1TriggerKeyExt_Stage2v0_hlt')
        ),
        cms.PSet(
            record = cms.string("L1TriggerKeyListExtRcd"),
            tag = cms.string("L1TriggerKeyListExt_Stage2v0_hlt")
        ),
        cms.PSet(
            record = cms.string("L1TUtmTriggerMenuRcd"),
            tag = cms.string("L1TUtmTriggerMenu_Stage2v0_hlt")
        )
    )
)

outputDB.DBParameters.authenticationPath = cms.untracked.string('.')
process.add_(outputDB)

process.l1mw  = cms.EDAnalyzer("L1MenuWriter")
process.l1kw  = cms.EDAnalyzer("L1KeyWriter")
process.l1klw = cms.EDAnalyzer("L1KeyListWriter")

process.p = cms.Path(process.l1kw + process.l1klw + process.l1mw)

