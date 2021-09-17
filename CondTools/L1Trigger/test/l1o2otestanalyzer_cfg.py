import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigValidation")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('runNumber',
                 4294967295, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number; default gives latest IOV")
options.register('inputDBConnect',
                 'sqlite_file:l1config.db', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Connection string for input DB")
options.register('inputDBAuth',
                 '.', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Authentication path for input DB")
options.register('printL1TriggerKeyList',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Print all object keys in CondDB")
options.register('printL1TriggerKey',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Print TSC key, subsystem keys, and object keys")
options.register('printRSKeys',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Print Run Settings keys")
options.register('printPayloadTokens',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Print payload tokens (long)")
options.parseArguments()

# Define CondDB tags
from CondTools.L1Trigger.L1CondEnum_cfi import L1CondEnum
from CondTools.L1Trigger.L1O2OTags_cfi import initL1O2OTags
initL1O2OTags()

# Input DB
from CondTools.L1Trigger.L1CondDBSource_cff import initCondDBSource
initCondDBSource( process,
                  inputDBConnect = options.inputDBConnect,
                  inputDBAuth = options.inputDBAuth,
                  tagBaseVec = initL1O2OTags.tagBaseVec,
                  includeRSTags = options.printRSKeys )

from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
outputDB = cms.Service("PoolDBOutputService",
                       CondDBSetup,
                       # BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                       connect = cms.string(options.inputDBConnect),
                       toPut = cms.VPSet(cms.PSet(
    record = cms.string("L1TriggerKeyRcd"),
    tag = cms.string("L1TriggerKey_" + initL1O2OTags.tagBaseVec[ L1CondEnum.L1TriggerKey ])),
                                         cms.PSet(
    record = cms.string("L1TriggerKeyListRcd"),
    tag = cms.string("L1TriggerKeyList_" + initL1O2OTags.tagBaseVec[ L1CondEnum.L1TriggerKeyList ]))
                                         ))
outputDB.DBParameters.authenticationPath = options.inputDBAuth

# PoolDBOutputService for printing out ESRecords
if options.printRSKeys == 1:
    from CondTools.L1Trigger.L1RSSubsystemParams_cfi import initL1RSSubsystems
    initL1RSSubsystems( tagBaseVec = initL1O2OTags.tagBaseVec )
    outputDB.toPut.extend(initL1RSSubsystems.params.recordInfo)

process.add_(outputDB)
    
# Source of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(options.runNumber),
    lastValue = cms.uint64(options.runNumber),
    interval = cms.uint64(1)
)

# Validation modules
process.load('CondTools.L1Trigger.L1O2OTestAnalyzer_cfi')
process.L1O2OTestAnalyzer.printPayloadTokens = False

if options.printL1TriggerKey == 1:
    process.L1O2OTestAnalyzer.printL1TriggerKey = True
else:
    process.L1O2OTestAnalyzer.printL1TriggerKey = False

if options.printL1TriggerKeyList == 1:
    process.L1O2OTestAnalyzer.printL1TriggerKeyList = True
else:
    process.L1O2OTestAnalyzer.printL1TriggerKeyList = False

if options.printRSKeys == 1:
    process.L1O2OTestAnalyzer.printESRecords = True
else:
    process.L1O2OTestAnalyzer.printESRecords = False

if options.printPayloadTokens == 1:
    process.L1O2OTestAnalyzer.printPayloadTokens = True
else:
    process.L1O2OTestAnalyzer.printPayloadTokens = False

process.p = cms.Path(process.L1O2OTestAnalyzer)
