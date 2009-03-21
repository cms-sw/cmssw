import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("L1ConfigWriteRSPayloadOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

options = VarParsing.VarParsing()
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
process.load("L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMaskAlgoTrigOnline_cfi")

# writer modules
process.load("CondTools.L1Trigger.L1CondDBPayloadWriter_cfi")
process.L1CondDBPayloadWriter.writeL1TriggerKey = cms.bool(False)
process.L1CondDBPayloadWriter.offlineDB = options.outputDBConnect
process.L1CondDBPayloadWriter.offlineAuthentication = options.outputDBAuth
process.L1CondDBPayloadWriter.L1TriggerKeyListTag = cms.string( 'L1TriggerKeyList_' + options.tagBase )

# Use highest possible run number so we always get the latest version
# of L1TriggerKeyList.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(4294967295),
    lastValue = cms.uint64(4294967295),
    interval = cms.uint64(1)
)

process.p = cms.Path(process.L1CondDBPayloadWriter)
