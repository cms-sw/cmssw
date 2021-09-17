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
options.register('tagUpdate',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Comma-separated list of column-separated pairs relating type to a new tagBase")
options.register('inputDBAuth',
                 '.', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Authentication path for input DB")
options.register('printL1TriggerKeyListExt',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Print all object keys in CondDB")
options.register('printL1TriggerKeyExt',
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
from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt
from CondTools.L1TriggerExt.L1O2OTagsExt_cfi import initL1O2OTagsExt
initL1O2OTagsExt()

# Override the tag bases if instructed to do so
if options.tagUpdate :
    for type2tagBase in options.tagUpdate.split(',') :
        (t,tagBase) = type2tagBase.split(':')
        index = L1CondEnumExt.__dict__[t]
        initL1O2OTagsExt.tagBaseVec[index] = tagBase

# Input DB
from CondTools.L1TriggerExt.L1CondDBSourceExt_cff import initCondDBSourceExt
initCondDBSourceExt( process,
                  inputDBConnect = options.inputDBConnect,
                  inputDBAuth = options.inputDBAuth,
                  tagBaseVec = initL1O2OTagsExt.tagBaseVec,
                  includeRSTags = options.printRSKeys )

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string(options.inputDBConnect)
outputDB = cms.Service("PoolDBOutputService",
                       CondDB,
                       # BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                       toPut = cms.VPSet(cms.PSet(
    record = cms.string("L1TriggerKeyExtRcd"),
    tag = cms.string("L1TriggerKeyExt_" + initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TriggerKeyExt ])),
                                         cms.PSet(
    record = cms.string("L1TriggerKeyListExtRcd"),
    tag = cms.string("L1TriggerKeyListExt_" + initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TriggerKeyListExt ]))
                                         ))
outputDB.DBParameters.authenticationPath = options.inputDBAuth

# PoolDBOutputService for printing out ESRecords
if options.printRSKeys == 1:
    from CondTools.L1TriggerExt.L1RSSubsystemParamsExt_cfi import initL1RSSubsystemsExt
    initL1RSSubsystemsExt( tagBaseVec = initL1O2OTagsExt.tagBaseVec )
    outputDB.toPut.extend(initL1RSSubsystemsExt.params.recordInfo)

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
process.load('CondTools.L1TriggerExt.L1O2OTestAnalyzerExt_cfi')
process.L1O2OTestAnalyzerExt.printPayloadTokens = False

if options.printL1TriggerKeyExt == 1:
    process.L1O2OTestAnalyzerExt.printL1TriggerKeyExt = True
else:
    process.L1O2OTestAnalyzerExt.printL1TriggerKeyExt = False

if options.printL1TriggerKeyListExt == 1:
    process.L1O2OTestAnalyzerExt.printL1TriggerKeyListExt = True
else:
    process.L1O2OTestAnalyzerExt.printL1TriggerKeyListExt = False

if options.printRSKeys == 1:
    process.L1O2OTestAnalyzerExt.printESRecords = True
else:
    process.L1O2OTestAnalyzerExt.printESRecords = False

if options.printPayloadTokens == 1:
    process.L1O2OTestAnalyzerExt.printPayloadTokens = True
else:
    process.L1O2OTestAnalyzerExt.printPayloadTokens = False

#print process.dumpPython()

process.p = cms.Path(process.L1O2OTestAnalyzerExt)
