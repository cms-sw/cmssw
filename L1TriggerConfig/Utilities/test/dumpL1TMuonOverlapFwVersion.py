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
#                 'TSCKEY_DUMMY:RSKEY_DUMMY',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "object key")
options.register('systemKey',
#                 '', # empty default value
                 'L1TMuonOverlapFwVersion_DUMMY',
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
                 "Authentication path for the DB")
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
    process.L1TriggerKeyOnlineExt.subsystemLabels = cms.vstring('OMTF')
    # include the system-specific subkeys ESProducer (generates OMTF labeled L1TriggerKey)
    process.load("L1TriggerConfig.L1TConfigProducers.L1TMuonOverlapObjectKeysOnline_cfi")
    process.L1TMuonOverlapObjectKeysOnline.onlineAuthentication = cms.string( options.DBAuth    )
    process.L1TMuonOverlapObjectKeysOnline.onlineDB             = cms.string( options.DBConnect )
else :
    # instantiate manually the system-specific L1TriggerKey using the subsystemKey option
    process.load("CondTools.L1TriggerExt.L1TriggerKeyDummyExt_cff")
    process.L1TriggerKeyDummyExt.tscKey = cms.string('TSCKEY_DUMMY')
    process.L1TriggerKeyDummyExt.objectKeys = cms.VPSet(
        cms.PSet(
            record = cms.string('L1TMuonOverlapFwVersionO2ORcd'),
            type = cms.string('L1TMuonOverlapFwVersion'),
            key = cms.string(options.systemKey)
        )
    )

# Online producer for the payload 
process.load("L1TriggerConfig.L1TConfigProducers.L1TMuonOverlapFwVersionOnline_cfi")
#process.load("L1Trigger.L1TMuonOverlap.fakeOmtfFwVersion_cff")
#process.L1TMuonOverlapFwVersionOnlineProd.onlineAuthentication = cms.string( options.DBAuth )
#process.L1TMuonOverlapFwVersionOnlineProd.onlineDB             = cms.string( options.DBConnect )
#
process.getter = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
       record = cms.string('L1TMuonOverlapFwVersionO2ORcd'),
#       record = cms.string('L1TMuonOverlapFwVersionRcd'),
       data   = cms.vstring('L1TMuonOverlapFwVersion')
   )),
   verbose = cms.untracked.bool(True)
)

process.l1mow = cms.EDAnalyzer("L1TMuonOverlapFwVersionTester", writeToDB = cms.untracked.bool(True), isO2Opayload = cms.untracked.bool(False))

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string(options.outputDBConnect)

outputDB = cms.Service("PoolDBOutputService",
    CondDB,
    toPut   = cms.VPSet(
        cms.PSet(
#            record = cms.string('L1TMuonOverlapFwVersionO2ORcd'),
            record = cms.string('L1TMuonOverlapFwVersionRcd'),
            tag = cms.string('L1TMuonOverlapFwVersion_Stage2v0_hlt')
        ),
        cms.PSet(
            record = cms.string("L1TriggerKeyListExtRcd"),
            tag = cms.string("L1TriggerKeyListExt_Stage2v0_hlt")
        )
    )
)

outputDB.DBParameters.authenticationPath = options.DBAuth
process.add_(outputDB)

process.p = cms.Path(process.getter + process.l1mow)

