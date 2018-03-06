from L1TriggerConfig.L1TConfigProducers.L1TUtmTriggerMenuOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonBarrelParamsOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonGlobalParamsOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonOverlapParamsOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonEndCapParamsOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonEndCapForestOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TCaloParamsOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TGlobalPrescalesVetosOnline_cfi import *

def setTSCPayloadsDB(process, DBConnect, DBAuth, protoDBConnect, protoDBAuth):

    process.L1TCaloParamsOnlineProd.onlineDB           = cms.string( DBConnect )
    process.L1TGlobalPrescalesVetosOnlineProd.onlineDB = cms.string( DBConnect )
    process.L1TMuonBarrelParamsOnlineProd.onlineDB     = cms.string( DBConnect )
    process.L1TMuonEndCapParamsOnlineProd.onlineDB     = cms.string( DBConnect )
    process.L1TMuonEndCapForestOnlineProd.onlineDB     = cms.string( DBConnect )
    process.L1TMuonGlobalParamsOnlineProd.onlineDB     = cms.string( DBConnect )
    process.L1TMuonOverlapParamsOnlineProd.onlineDB    = cms.string( DBConnect )
    process.L1TUtmTriggerMenuOnlineProd.onlineDB       = cms.string( DBConnect )

    process.L1TCaloParamsOnlineProd.onlineAuthentication           = cms.string( DBAuth )
    process.L1TGlobalPrescalesVetosOnlineProd.onlineAuthentication = cms.string( DBAuth )
    process.L1TMuonBarrelParamsOnlineProd.onlineAuthentication     = cms.string( DBAuth )
    process.L1TMuonEndCapParamsOnlineProd.onlineAuthentication     = cms.string( DBAuth )
    process.L1TMuonEndCapForestOnlineProd.onlineAuthentication     = cms.string( DBAuth )
    process.L1TMuonGlobalParamsOnlineProd.onlineAuthentication     = cms.string( DBAuth )
    process.L1TMuonOverlapParamsOnlineProd.onlineAuthentication    = cms.string( DBAuth )
    process.L1TUtmTriggerMenuOnlineProd.onlineAuthentication       = cms.string( DBAuth )

    process.l1caloparProtodb.connect                         = cms.string( protoDBConnect )
    process.l1bmtfparProtodb.connect                         = cms.string( protoDBConnect )
    process.l1emtfparProtodb.connect                         = cms.string( protoDBConnect )
    process.l1gmtparProtodb.connect                          = cms.string( protoDBConnect )
    process.l1caloparProtodb.DBParameters.authenticationPath = cms.untracked.string( protoDBAuth )
    process.l1bmtfparProtodb.DBParameters.authenticationPath = cms.untracked.string( protoDBAuth )
    process.l1emtfparProtodb.DBParameters.authenticationPath = cms.untracked.string( protoDBAuth )
    process.l1gmtparProtodb.DBParameters.authenticationPath  = cms.untracked.string( protoDBAuth )

def liftPayloadSafetyFor(process, systems):
    if 'CALO' in systems:
        process.L1TCaloParamsOnlineProd.transactionSafe           = cms.bool(False)

    if 'uGTrs' in systems:
        process.L1TGlobalPrescalesVetosOnlineProd.transactionSafe = cms.bool(False)

    if 'BMTF' in systems:
        process.L1TMuonBarrelParamsOnlineProd.transactionSafe     = cms.bool(False)

    if 'EMTF' in systems:
        process.L1TMuonEndCapParamsOnlineProd.transactionSafe     = cms.bool(False)
        process.L1TMuonEndCapForestOnlineProd.transactionSafe     = cms.bool(False)

    if 'uGMT' in systems:
        process.L1TMuonGlobalParamsOnlineProd.transactionSafe     = cms.bool(False)

    if 'OMTF' in systems:
        process.L1TMuonOverlapParamsOnlineProd.transactionSafe    = cms.bool(False)

