from L1TriggerConfig.L1TConfigProducers.L1TUtmTriggerMenuObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonBarrelObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonGlobalObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonOverlapObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonEndCapObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TCaloParamsObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TGlobalPrescalesVetosObjectKeysOnline_cfi import *

def setTSCKeysDB(process, DBConnect, DBAuth):

    process.L1TCaloParamsObjectKeysOnline.onlineDB           = cms.string( DBConnect )
    process.L1TGlobalPrescalesVetosObjectKeysOnline.onlineDB = cms.string( DBConnect )
    process.L1TMuonBarrelObjectKeysOnline.onlineDB           = cms.string( DBConnect )
    process.L1TMuonEndCapObjectKeysOnline.onlineDB           = cms.string( DBConnect )
    process.L1TMuonGlobalObjectKeysOnline.onlineDB           = cms.string( DBConnect )
    process.L1TMuonOverlapObjectKeysOnline.onlineDB          = cms.string( DBConnect )
    process.L1TUtmTriggerMenuObjectKeysOnline.onlineDB       = cms.string( DBConnect )

    process.L1TCaloParamsObjectKeysOnline.onlineAuthentication           = cms.string( DBAuth )
    process.L1TGlobalPrescalesVetosObjectKeysOnline.onlineAuthentication = cms.string( DBAuth )
    process.L1TMuonBarrelObjectKeysOnline.onlineAuthentication           = cms.string( DBAuth )
    process.L1TMuonEndCapObjectKeysOnline.onlineAuthentication           = cms.string( DBAuth )
    process.L1TMuonGlobalObjectKeysOnline.onlineAuthentication           = cms.string( DBAuth )
    process.L1TMuonOverlapObjectKeysOnline.onlineAuthentication          = cms.string( DBAuth )
    process.L1TUtmTriggerMenuObjectKeysOnline.onlineAuthentication       = cms.string( DBAuth )


def liftKeySafetyFor(process, systems):
    if 'CALO' in systems:
        process.L1TCaloParamsObjectKeysOnline.transactionSafe           = cms.bool(False)

    if 'uGTrs' in systems:
        process.L1TGlobalPrescalesVetosObjectKeysOnline.transactionSafe = cms.bool(False)

    if 'BMTF' in systems:
        process.L1TMuonBarrelObjectKeysOnline.transactionSafe           = cms.bool(False)

    if 'EMTF' in systems:
        process.L1TMuonEndCapObjectKeysOnline.transactionSafe           = cms.bool(False)

    if 'uGMT' in systems:
        process.L1TMuonGlobalObjectKeysOnline.transactionSafe           = cms.bool(False)

    if 'OMTF' in systems:
        process.L1TMuonOverlapObjectKeysOnline.transactionSafe          = cms.bool(False)
