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
    process.L1TriggerKeyOnlineExt.subsystemLabels = cms.vstring('BMTF')
    # include the system-specific subkeys ESProducer (generates BMTF labeled L1TriggerKey)
    process.load("L1TriggerConfig.L1TConfigProducers.L1TMuonBarrelObjectKeysOnline_cfi")
    process.L1TMuonBarrelObjectKeysOnline.onlineAuthentication = cms.string( options.DBAuth    )
    process.L1TMuonBarrelObjectKeysOnline.onlineDB             = cms.string( options.DBConnect )
else :
    # instantiate manually the system-specific L1TriggerKey using the subsystemKey option
    process.load("CondTools.L1TriggerExt.L1TriggerKeyDummyExt_cff")
    process.L1TriggerKeyDummyExt.tscKey = cms.string('dummyL1TMuonBarrel')
    process.L1TriggerKeyDummyExt.objectKeys = cms.VPSet(
        cms.PSet(
            record = cms.string('L1TMuonBarrelParamsO2ORcd'),
            type = cms.string('L1TMuonBarrelParams'),
            key = cms.string(options.systemKey)
        )
    )

# Online produced for the payload 
process.load("L1TriggerConfig.L1TConfigProducers.L1TMuonBarrelParamsOnline_cfi")
process.L1TMuonBarrelParamsOnlineProd.onlineAuthentication = cms.string( options.DBAuth )
process.L1TMuonBarrelParamsOnlineProd.onlineDB             = cms.string( options.DBConnect )


process.getter = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
       record = cms.string('L1TMuonBarrelParamsO2ORcd'),
       data   = cms.vstring('L1TMuonBarrelParams')
   )),
   verbose = cms.untracked.bool(True)
)

process.l1mbw = cms.EDAnalyzer("L1TMuonBarrelParamsWriter")

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string(options.outputDBConnect)

outputDB = cms.Service("PoolDBOutputService",
    CondDB,
    toPut   = cms.VPSet(
        cms.PSet(
            record = cms.string('L1TMuonBarrelParamsO2ORcd'),
            tag = cms.string('L1TMuonBarrelParams_Stage2v0_hlt')
        ),
        cms.PSet(
            record = cms.string("L1TriggerKeyListExtRcd"),
            tag = cms.string("L1TriggerKeyListExt_Stage2v0_hlt")
        )
    )
)

outputDB.DBParameters.authenticationPath = options.DBAuth
process.add_(outputDB)

process.p = cms.Path(process.getter + process.l1mbw)

