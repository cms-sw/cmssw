import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
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
options.register('startup',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Use L1StartupConfig_cff instead of L1DummyConfig_cff")
options.parseArguments()

# Generate dummy L1TriggerKey
process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")

# Generate dummy configuration data
if options.startup == 0:
    process.load("L1Trigger.Configuration.L1DummyConfig_cff")
    process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu_MC2009_v4_L1T_Scales_20090624_Imp0_Unprescaled_cff")
else:
    process.load("L1Trigger.Configuration.L1StartupConfig_cff")
    process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_Commissioning2009_v5_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff")

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
from CondTools.L1Trigger.L1CondDBPayloadWriter_cff import initPayloadWriter
initPayloadWriter( process,
                   outputDBConnect = options.outputDBConnect,
                   outputDBAuth = options.outputDBAuth,
                   tagBaseVec = tagBaseVec )

# Generate dummy L1TriggerKeyList
process.L1CondDBPayloadWriter.newL1TriggerKeyList = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.p = cms.Path(process.L1CondDBPayloadWriter)
#process.l1CSCTFConfig.ptLUT_path = '/afs/cern.ch/cms/MUON/csc/fast1/track_finder/luts/PtLUT.dat'
process.l1CSCTFConfig.ptLUT_path = './PtLUT.dat'
