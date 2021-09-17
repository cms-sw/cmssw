import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('tagBase',
                 'IDEAL', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "IOV tags = object_{tagBase}")
options.register('useO2OTags',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "0 = use uniform tags, 1 = ignore tagBase and use O2O tags")
options.register('outputDBConnect',
                 'sqlite_file:l1config.db', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Connection string for output DB")
options.register('outputDBAuth',
                 '.', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Authentication path for output DB")
options.parseArguments()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

# Generate dummy L1TriggerKey
process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.L1TriggerKeyDummy.objectKeys = cms.VPSet()
process.L1TriggerKeyDummy.tscKey = cms.string(' ')

# Define CondDB tags
if options.useO2OTags == 0:
    from CondTools.L1Trigger.L1CondEnum_cfi import L1CondEnum
    from CondTools.L1Trigger.L1UniformTags_cfi import initL1UniformTags
    initL1UniformTags( tagBase = options.tagBase )
    tagBaseVec = initL1UniformTags.tagBaseVec
else:
    from CondTools.L1Trigger.L1CondEnum_cfi import L1CondEnum
    from CondTools.L1Trigger.L1O2OTags_cfi import initL1O2OTags
    initL1O2OTags()
    tagBaseVec = initL1O2OTags.tagBaseVec

# writer modules
from CondTools.L1Trigger.L1CondDBPayloadWriter_cff import initPayloadWriter
initPayloadWriter( process,
                   outputDBConnect = options.outputDBConnect,
                   outputDBAuth = options.outputDBAuth,
                   tagBaseVec = tagBaseVec )

# Generate dummy L1TriggerKeyList to initialize DB on the first time ONLY.
process.L1CondDBPayloadWriter.newL1TriggerKeyList = True

process.p = cms.Path(process.L1CondDBPayloadWriter)
