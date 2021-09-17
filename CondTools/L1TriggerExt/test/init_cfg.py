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
process.load("CondTools.L1TriggerExt.L1TriggerKeyDummyExt_cff")
process.L1TriggerKeyDummyExt.objectKeys = cms.VPSet()
process.L1TriggerKeyDummyExt.tscKey = cms.string(' ')

# Define CondDB tags
if options.useO2OTags == 0:
    from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt
    from CondTools.L1TriggerExt.L1UniformTagsExt_cfi import initL1UniformTagsExt
    initL1UniformTagsExt( tagBase = options.tagBase )
    tagBaseVec = initL1UniformTagsExt.tagBaseVec
else:
    from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt
    from CondTools.L1TriggerExt.L1O2OTagsExt_cfi import initL1O2OTagsExt
    initL1O2OTagsExt()
    tagBaseVec = initL1O2OTagsExt.tagBaseVec

# writer modules
from CondTools.L1TriggerExt.L1CondDBPayloadWriterExt_cff import initPayloadWriterExt
initPayloadWriterExt( process,
                   outputDBConnect = options.outputDBConnect,
                   outputDBAuth = options.outputDBAuth,
                   tagBaseVec = tagBaseVec )

# Generate dummy L1TriggerKeyListExt to initialize DB on the first time ONLY.
process.L1CondDBPayloadWriterExt.newL1TriggerKeyListExt = True

process.p = cms.Path(process.L1CondDBPayloadWriterExt)
