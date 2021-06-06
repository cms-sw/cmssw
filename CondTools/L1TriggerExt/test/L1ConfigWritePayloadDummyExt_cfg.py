import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
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
options.register('startup',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Use L1StartupConfig_cff instead of L1DummyConfig_cff")
options.parseArguments()

# Generate dummy L1TriggerKey
process.load("CondTools.L1TriggerExt.L1TriggerKeyDummyExt_cff")

# Generate dummy configuration data
if options.startup == 0:
    process.load("L1Trigger.Configuration.L1DummyConfig_cff")
    process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu_MC2009_v4_L1T_Scales_20090624_Imp0_Unprescaled_cff")
else:
    process.load("L1Trigger.Configuration.L1StartupConfig_cff")
    process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_Commissioning2009_v5_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff")

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

# Generate dummy L1TriggerKeyListExt
process.L1CondDBPayloadWriterExt.newL1TriggerKeyListExt = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.p = cms.Path(process.L1CondDBPayloadWriterExt)
