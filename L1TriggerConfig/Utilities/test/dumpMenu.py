import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

# Generate dummy L1TriggerKeyList
process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")
process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.L1TriggerKeyDummy.objectKeys = cms.VPSet(cms.PSet(
    record = cms.string("L1TUtmTriggerMenuRcd"),
    type   = cms.string("L1TUtmTriggerMenu"),
    key    = cms.string("dummy")
))


process.load("L1Trigger.L1TGlobal.TriggerMenu_cff")
process.TriggerMenu.L1TriggerMenuFile = cms.string('L1Menu_Collisions2016_dev_v4.xml')

process.getter = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(
       cms.PSet(
           record = cms.string('L1TUtmTriggerMenuRcd'),
           data   = cms.vstring('L1TUtmTriggerMenu')
       )
   ),
   verbose = cms.untracked.bool(True)
)

process.l1mw = cms.EDAnalyzer("L1MenuWriter", isO2Opayload = cms.untracked.bool(False))

from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
outputDB = cms.Service("PoolDBOutputService",
                       CondDBSetup,
                       connect = cms.string('sqlite_file:L1Menu_Collisions2016_dev_v4.xml.db'),
                       toPut   = cms.VPSet(
                           cms.PSet(
                               record = cms.string('L1TUtmTriggerMenuRcd'),
                               tag = cms.string("L1Menu_Collisions2016_dev_v4")
                           )
                       )
)
outputDB.DBParameters.authenticationPath = '.'
process.add_(outputDB)

process.p = cms.Path(process.getter + process.l1mw)
