# Example using L1RCTParameters

import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('objectKey',
                 'dummy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "object key")
options.register('objectType',
                 'L1TMuonOverlapParams', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "object C++ type")
options.register('recordName',
                 'L1TMuonOverlapParamsO2ORcd', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Name of EventSetup record")
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
options.register('genFromOMDS',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "0 = use dummy payloads, 1 = generate payloads from OMDS")
options.register('outputDBConnect',
                 'sqlite_file:l1config.db', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Connection string for output DB")
options.register('outputDBAuth',
                 '.', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Authentication path for outputDB")
options.register('overwriteKey',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Overwrite existing key")
options.register('startup',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Use L1StartupConfig_cff instead of L1DummyConfig_cff")

options.parseArguments()

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

# Generate L1TriggerKey
process.load("CondTools.L1TriggerExt.L1TriggerKeyDummyExt_cff")
process.L1TriggerKeyDummyExt.objectKeys = cms.VPSet(cms.PSet(
    record = cms.string(options.recordName),
    type = cms.string(options.objectType),
    key = cms.string(options.objectKey)
))

# Get L1TriggerKeyListExt from DB
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = cms.string(options.outputDBConnect)

process.outputDB = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListExtRcd'),
        tag = cms.string('L1TriggerKeyListExt_' + tagBaseVec[ L1CondEnumExt.L1TriggerKeyListExt ])
    ))
)

process.es_prefer_outputDB = cms.ESPrefer("PoolDBESSource","outputDB")
process.outputDB.DBParameters.authenticationPath = cms.untracked.string(options.outputDBAuth)

if options.genFromOMDS == 0:
    # Generate dummy configuration data
    process.load('L1TriggerConfig.L1TConfigProducers.L1TMuonOverlapParamsOnlineProxy_cfi')
    process.load('L1TriggerConfig.L1TConfigProducers.L1TMuonEndcapParamsOnlineProxy_cfi')
#    if options.startup == 0:
#        process.load("L1Trigger.Configuration.L1DummyConfig_cff")
#        process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu_MC2009_v2_L1T_Scales_20090624_Imp0_Unprescaled_cff")
#    else:
#        process.load("L1Trigger.Configuration.L1StartupConfig_cff")
#        process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_Commissioning2009_v3_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff")

else:
    # Generate configuration data from OMDS
    process.load("CondTools.L1TriggerExt.L1ConfigTSCPayloadsExt_cff")
    process.load("CondTools.L1TriggerExt.L1ConfigRSPayloadsExt_cff")

# writer modules
from CondTools.L1TriggerExt.L1CondDBPayloadWriterExt_cff import initPayloadWriterExt
initPayloadWriterExt( process,
                   outputDBConnect = options.outputDBConnect,
                   outputDBAuth = options.outputDBAuth,
                   tagBaseVec = tagBaseVec )
process.L1CondDBPayloadWriterExt.writeL1TriggerKeyExt = False

if options.overwriteKey == 0:
    process.L1CondDBPayloadWriterExt.overwriteKeys = False
else:
    process.L1CondDBPayloadWriterExt.overwriteKeys = True
#    if options.genFromOMDS != 0:

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.p = cms.Path(process.L1CondDBPayloadWriterExt)
