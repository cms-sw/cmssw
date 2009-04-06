import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWriteRSPayloadOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('runNumber',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number")
options.register('tagBase',
                 'CRAFT_hlt', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "IOV tags = object_{tagBase}")
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

process.load("CondTools.L1Trigger.L1ConfigRSKeys_cff")

# Get L1TriggerKeyList from DB
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.outputDB = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListRcd'),
        tag = cms.string('L1TriggerKeyList_' + options.tagBase )
    ))
)
#process.es_prefer_outputDB = cms.ESPrefer("PoolDBESSource","outputDB")
process.outputDB.connect = options.outputDBConnect
process.outputDB.DBParameters.authenticationPath = options.outputDBAuth

# Generate configuration data
process.load("L1TriggerConfig.RCTConfigProducers.L1RCTChannelMaskOnline_cfi")
process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTChannelMaskConfigOnline_cfi")
process.load("L1TriggerConfig.L1GtConfigProducers.l1GtPrescaleFactorsAlgoTrigOnline_cfi")
process.load("L1TriggerConfig.L1GtConfigProducers.l1GtPrescaleFactorsTechTrigOnline_cfi")
process.load("L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMaskAlgoTrigOnline_cfi")
process.load("L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMaskTechTrigOnline_cfi")
process.load("L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMaskVetoTechTrigOnline_cfi")

# writer modules
from CondTools.L1Trigger.L1CondDBPayloadWriter_cff import initPayloadWriter
initPayloadWriter( process,
                   outputDBConnect = options.outputDBConnect,
                   outputDBAuth = options.outputDBAuth,
                   tagBase = options.tagBase )
process.L1CondDBPayloadWriter.writeL1TriggerKey = cms.bool(False)

from CondTools.L1Trigger.L1CondDBIOVWriter_cff import initIOVWriter
initIOVWriter( process,
               outputDBConnect = options.outputDBConnect,
               outputDBAuth = options.outputDBAuth,
               tagBase = options.tagBase,
               tscKey = '' )
initIOVWriter.outputDB.toPut.extend( cms.VPSet(cms.PSet(
    record = cms.string("L1TriggerKeyListRcd"),
    tag = cms.string("L1TriggerKeyList_" + options.tagBase))) )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(options.runNumber),
    lastValue = cms.uint64(options.runNumber),
    interval = cms.uint64(1)
)

process.p = cms.Path(process.L1CondDBPayloadWriter*process.L1CondDBIOVWriter)
