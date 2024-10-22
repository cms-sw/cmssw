# Example using L1RCTParameters

import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWriteIOVDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('objectKey',
                 'dummy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('objectType',
                 'L1RCTParameters', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "object C++ type")
options.register('recordName',
                 'L1RCTParametersRcd', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Name of EventSetup record")
options.register('tagName',
                 'L1RCTParameters', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "IOV tags = {tagName}_{tagBase}")
options.register('useO2OTags',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "0 = use uniform tags, 1 = ignore tagBase and use O2O tags")
options.register('condIndex',
                 -999, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Index in L1CondEnum of record")
options.register('runNumber',
                 1000, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number")
options.register('tagBase',
                 'IDEAL', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "IOV tags = {tagName}_{tagBase}")
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
options.parseArguments()

# Define CondDB tags
if options.useO2OTags == 0:
    from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt
    from CondTools.L1TriggerExt.L1UniformTagsExt_cfi import initL1UniformTagsExt
    initL1UniformTagsExt( tagBase = options.tagBase )
    tagBaseVec = initL1UniformTagsExt.tagBaseVec
    options.condIndex = 0 # Doesn't matter what index is used with uniform tags
else:
    from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt
    from CondTools.L1TriggerExt.L1O2OTagsExt_cfi import initL1O2OTagsExt
    initL1O2OTagsExt()
    tagBaseVec = initL1O2OTagsExt.tagBaseVec

# Get L1TriggerKeyListExt from DB
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.outputDB = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListExtRcd'),
        tag = cms.string('L1TriggerKeyListExt_' + tagBaseVec[ L1CondEnumExt.L1TriggerKeyListExt ])
    ))
)
process.outputDB.connect = cms.string(options.outputDBConnect)
process.outputDB.DBParameters.authenticationPath = options.outputDBAuth

# writer modules
from CondTools.L1TriggerExt.L1CondDBIOVWriterExt_cff import initIOVWriterExt
initIOVWriterExt( process,
               outputDBConnect = options.outputDBConnect,
               outputDBAuth = options.outputDBAuth,
               tagBaseVec = tagBaseVec,
               tscKey = options.objectKey )
process.L1CondDBIOVWriterExt.ignoreTriggerKey = cms.bool(True)
process.L1CondDBIOVWriterExt.toPut = cms.VPSet(cms.PSet(
    record = cms.string(options.recordName),
    type = cms.string(options.objectType),
    tag = cms.string(options.tagName + '_' + tagBaseVec[ options.condIndex ])
))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(options.runNumber),
    lastValue = cms.uint64(options.runNumber),
    interval = cms.uint64(1)
)

process.p = cms.Path(process.L1CondDBIOVWriterExt)
