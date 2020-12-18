'''
Example on how to use this script:
1) prepare the payload with L1ConfigWriteSinglePayloadExt_cfg.py
   (as, for example, is done in L1TriggerConfig/Utilities/test/runOneByOne.sh)
2) call this script with first run of the new IOV and system key used above
   e.g.:
   cmsRun L1ConfigWriteIOVDummyExt_cfg.py runNumber=1 tscKey='EMTF_ALGO_EMPTY' useO2OTags=1
'''
import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWriteIOVDummyExt")
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

# Get L1TriggerKeyListExt from DB

# Define CondDB tags
if options.useO2OTags == 0:
    from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt
    from CondTools.L1TriggerExt.L1UniformTagsExt_cfi import initL1UniformTagsExt
    initL1UniformTagsExt( tagBase = options.tagBase )
    tagBaseVec = initL1UniformTagsExt.tagBaseVec
else:
    from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt
    from CondTools.L1TriggerExt.L1O2OTagsExt_cfi import initL1O2OTagsExt
    initL1O2OTagsExt()
    tagBaseVec = initL1O2OTagsExt.tagBaseVec

#
process.load("CondTools.L1TriggerExt.L1TriggerKeyDummyExt_cff")
process.L1TriggerKeyDummyExt.tscKey = cms.string('dummyTransientKey_' + options.tscKey)
process.L1TriggerKeyDummyExt.objectKeys = cms.VPSet(
    cms.PSet(
        record = cms.string('L1TMuonEndCapParamsO2ORcd'),
        type = cms.string('L1TMuonEndCapParams'),
        key = cms.string(options.tscKey)
    )
)

    
# writer modules
from CondTools.L1TriggerExt.L1CondDBIOVWriterExt_cff import initIOVWriterExt
initIOVWriterExt( process,
               outputDBConnect = options.outputDBConnect,
               outputDBAuth = options.outputDBAuth,
               tagBaseVec = tagBaseVec,
               tscKey = '' )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(options.runNumber),
    lastValue = cms.uint64(options.runNumber),
    interval = cms.uint64(1)
)

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = cms.string(options.outputDBConnect)

process.outputDB = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListExtRcd'),
        tag = cms.string('L1TriggerKeyListExt_' + tagBaseVec[ L1CondEnumExt.L1TriggerKeyListExt ])
    ))
)

process.outputDB.DBParameters.authenticationPath = options.outputDBAuth

process.p = cms.Path(process.L1CondDBIOVWriterExt)
