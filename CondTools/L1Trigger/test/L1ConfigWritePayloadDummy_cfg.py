import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('tagBase',
                 'IDEAL', #default value
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
                 "Authentication path for output DB")
options.parseArguments()

# Generate dummy L1TriggerKey and L1TriggerKeyList
process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")

# Generate dummy configuration data
process.load("L1Trigger.Configuration.L1DummyConfig_cff")

# writer modules
from CondTools.L1Trigger.L1CondDBPayloadWriter_cff import initPayloadWriter
initPayloadWriter( process,
                   outputDBConnect = options.outputDBConnect,
                   outputDBAuth = options.outputDBAuth,
                   tagBase = options.tagBase )

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
process.l1CSCTFConfig.ptLUT_path = '/afs/cern.ch/cms/MUON/csc/fast1/track_finder/luts/PtLUT.dat'
