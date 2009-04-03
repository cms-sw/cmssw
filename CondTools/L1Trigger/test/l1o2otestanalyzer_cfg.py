import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigValidation")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('runNumber',
                 4294967295, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number; default gives latest IOV")
options.register('tagBase',
                 'IDEAL', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "IOV tags = object_{tagBase}")
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
options.register('use30XTagList',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Set to 1 for conditions written in 30X")
options.register('printL1TriggerKeyList',
                 1, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Self explanatory")
options.register('printL1TriggerKey',
                 1, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Self explanatory")
options.parseArguments()

# Input DB
from CondTools.L1Trigger.L1CondDBSource_cff import initCondDBSource
initCondDBSource( process,
                  inputDBConnect = options.inputDBConnect,
                  inputDBAuth = options.inputDBAuth,
                  tagBase = options.tagBase,
                  use30XTagList = options.use30XTagList )

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

process.p = cms.Path(process.L1O2OTestAnalyzer)
