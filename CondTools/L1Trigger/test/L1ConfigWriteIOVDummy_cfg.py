import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWriteIOVDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('tscKey',
                 'dummy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "TSC key")
options.register('runNumber',
                 1000, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number")
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

# Get L1TriggerKeyList from DB
process.load("CondCore.DBCommon.CondDBCommon_cfi")

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
from CondTools.L1Trigger.L1CondDBIOVWriter_cff import initIOVWriter
initIOVWriter( process,
               outputDBConnect = options.outputDBConnect,
               outputDBAuth = options.outputDBAuth,
               tagBaseVec = tagBaseVec,
               tscKey = options.tscKey )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(options.runNumber),
    lastValue = cms.uint64(options.runNumber),
    interval = cms.uint64(1)
)

process.outputDB = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListRcd'),
        tag = cms.string('L1TriggerKeyList_' + tagBaseVec[ L1CondEnum.L1TriggerKeyList ])
    ))
)

process.p = cms.Path(process.L1CondDBIOVWriter)
process.outputDB.connect = cms.string(options.outputDBConnect)
process.outputDB.DBParameters.authenticationPath = options.outputDBAuth
