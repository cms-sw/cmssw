from __future__ import print_function
# to test the communication with DBS and produce the csctf configuration
import FWCore.ParameterSet.Config as cms

process = cms.Process("QWE")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('topKey',
                 '', # empty default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "object key")
options.register('systemKey',
                 '', # empty default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "object key")
options.register('outputDBConnect',
                 'sqlite_file:./l1config.db', # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Connection string for output DB")
options.register('DBConnect',
                 'oracle://cms_omds_adg/CMS_TRG_R', # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "OMDS connect string")
options.register('DBAuth',
                 '.', # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Authentication path for DB")
options.parseArguments()

# sanity checks
if ( len(options.topKey) and len(options.systemKey) ) or ( len(options.topKey)==0 and len(options.systemKey)==0 ) :
    print("Specify either the topKey (top-level tsc:rs key) or systemKey (system specific tsc:rs key), but not both")
    exit(1)

# standard CMSSW stuff
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

# add dummy L1TriggerKeyList so as to suppress framework related warning
process.load("CondTools.L1TriggerExt.L1TriggerKeyListDummyExt_cff")

# produce L1TriggerKey for the subsystem online producers
if len(options.topKey) :
    # parent L1TriggerKey that will seed system-specific key to be automatically generated below
    process.load("CondTools.L1TriggerExt.L1TriggerKeyRcdSourceExt_cfi")
    process.load("CondTools.L1TriggerExt.L1SubsystemKeysOnlineExt_cfi")
    process.L1SubsystemKeysOnlineExt.tscKey = cms.string( options.topKey.split(':')[0] )
    process.L1SubsystemKeysOnlineExt.rsKey  = cms.string( options.topKey.split(':')[1] )
    process.L1SubsystemKeysOnlineExt.onlineAuthentication = cms.string( options.DBAuth )
    process.L1SubsystemKeysOnlineExt.forceGeneration = cms.bool(True)
    # using the parent L1TriggerKey above start generation of system-specific (labeled) L1TriggerKeys and pack them the main (unlabeled) L1TriggerKey (just one subsystem here)
    process.load("CondTools.L1TriggerExt.L1TriggerKeyOnlineExt_cfi")
    process.L1TriggerKeyOnlineExt.subsystemLabels = cms.vstring('uGT')
    # include the subsystem-specific subkeys ESProducer (generates uGT labeled L1TriggerKey)
    process.load("L1TriggerConfig.L1TConfigProducers.L1TUtmTriggerMenuObjectKeysOnline_cfi")
    process.L1TUtmTriggerMenuObjectKeysOnline.onlineAuthentication = cms.string( options.DBAuth    )
    process.L1TUtmTriggerMenuObjectKeysOnline.onlineDB             = cms.string( options.DBConnect )
else :
    # generate the parent L1TriggerKey 
    process.load("CondTools.L1TriggerExt.L1TriggerKeyDummyExt_cff")
    process.L1TriggerKeyDummyExt.tscKey = cms.string('dummyL1TUtmTriggerMenu')
    process.L1TriggerKeyDummyExt.label  = cms.string('SubsystemKeysOnly')
    process.L1TriggerKeyDummyExt.uGTKey = cms.string(options.systemKey)
    # using the parent L1TriggerKey trigger generation of system specific (labeled) L1TriggerKeys and pack them the main (unlabeled) L1TriggerKey (just one subsystem here)
    process.load("CondTools.L1TriggerExt.L1TriggerKeyOnlineExt_cfi")
    process.L1TriggerKeyOnlineExt.subsystemLabels = cms.vstring('uGT')
    # include the uGT specific key ESProducer (generates uGT labeled L1TriggerKey) and the corresponding payload ESProduced
    process.load("L1TriggerConfig.L1TConfigProducers.L1TUtmTriggerMenuObjectKeysOnline_cfi")
    process.L1TUtmTriggerMenuObjectKeysOnline.onlineAuthentication = cms.string( options.DBAuth )

# Online produced for the payload 
process.load("L1TriggerConfig.L1TConfigProducers.L1TUtmTriggerMenuOnline_cfi")
process.L1TUtmTriggerMenuOnlineProd.onlineAuthentication = cms.string( options.DBAuth )
process.L1TUtmTriggerMenuOnlineProd.onlineDB             = cms.string( options.DBConnect )

#process.l1cr = cms.EDAnalyzer( "L1TriggerKeyExtReader", label = cms.string("uGT") )
## label = cms.string("SubsystemKeysOnly") )

process.getter = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
       record = cms.string('L1TUtmTriggerMenuO2ORcd'),
       data   = cms.vstring('L1TUtmTriggerMenu')
   )),
   verbose = cms.untracked.bool(True)
)

process.l1mw   = cms.EDAnalyzer("L1MenuWriter", isO2Opayload = cms.untracked.bool(True) )
process.l1tkw  = cms.EDAnalyzer("L1KeyWriter")
process.l1tklw = cms.EDAnalyzer("L1KeyListWriter")

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string(options.outputDBConnect)

outputDB = cms.Service("PoolDBOutputService",
    CondDB,
    toPut   = cms.VPSet(
        cms.PSet(
            record = cms.string('L1TUtmTriggerMenuO2ORcd'),
            tag = cms.string('L1TUtmTriggerMenu_Stage2v0_hlt')
        ),
        cms.PSet(
            record = cms.string("L1TriggerKeyListExtRcd"),
            tag = cms.string("L1TriggerKeyListExt_Stage2v0_hlt")
        ),
#        cms.PSet(
#            record = cms.string("L1TriggerKeyExtRcd"),
#            tag = cms.string("L1TriggerKeyExt_Stage2v0_hlt")
#        )
    )
)

outputDB.DBParameters.authenticationPath = options.DBAuth
process.add_(outputDB)

process.load('CondTools.L1TriggerExt.L1CondDBPayloadWriterExt_cfi')

#process.p = cms.Path(process.l1cr)
process.p = cms.Path(process.getter + process.l1mw) #+ process.l1tkw + process.l1tklw  + process.L1CondDBPayloadWriterExt)

