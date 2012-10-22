import FWCore.ParameterSet.Config as cms

# options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('globalTag',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Global Tag")
options.register('sqlite',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "SQLite file")
options.register('run',
                 '1', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "run number")
options.register('cfi',
                 '', # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "CMSSW cfi file")
                 
options.parseArguments()


print "Global Tag      : ", options.globalTag
print "SQLite          : ", options.sqlite
print "Fake Conditions : ", options.cfi
print "Run             : ", options.run


# the job
process = cms.Process("L1GctConfigDump")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('l1GctConfigDump')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(options.run),
    lastValue = cms.uint64(options.run),
    interval = cms.uint64(1)
)

if (options.globalTag != "") :
    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
    process.GlobalTag.globaltag = options.globalTag+"::All"

if (options.sqlite != "") :
    process.load("CondTools.L1Trigger.L1CondDBSource_cff")
    print "Can't read SQLite files yet"

if (options.cfi != "") :
    process.load("L1Trigger.Configuration.L1Trigger_FakeConditions_cff")

#from CondCore.DBCommon.CondDBSetup_cfi import *


process.load("L1TriggerConfig.GctConfigProducers.l1GctConfigDump_cfi")

process.path = cms.Path(
    process.l1GctConfigDump
)

