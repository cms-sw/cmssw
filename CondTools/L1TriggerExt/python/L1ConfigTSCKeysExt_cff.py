from L1TriggerConfig.L1TConfigProducers.L1TUtmTriggerMenuObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonBarrelObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonGlobalObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonOverlapObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonEndcapObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TCaloParamsObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TGlobalPrescalesVetosObjectKeysOnline_cfi import *

def setTSCKeysDB(process, DBConnect, DBAuth):

    process.L1TCaloParamsObjectKeysOnline.onlineDB           = cms.string( DBConnect )
    process.L1TGlobalPrescalesVetosObjectKeysOnline.onlineDB = cms.string( DBConnect )
    process.L1TMuonBarrelObjectKeysOnline.onlineDB           = cms.string( DBConnect )
    process.L1TMuonEndcapObjectKeysOnline.onlineDB           = cms.string( DBConnect )
    process.L1TMuonGlobalObjectKeysOnline.onlineDB           = cms.string( DBConnect )
    process.L1TMuonOverlapObjectKeysOnline.onlineDB          = cms.string( DBConnect )
    process.L1TUtmTriggerMenuObjectKeysOnline.onlineDB       = cms.string( DBConnect )

    process.L1TCaloParamsObjectKeysOnline.onlineAuthentication           = cms.string( DBAuth )
    process.L1TGlobalPrescalesVetosObjectKeysOnline.onlineAuthentication = cms.string( DBAuth )
    process.L1TMuonBarrelObjectKeysOnline.onlineAuthentication           = cms.string( DBAuth )
    process.L1TMuonEndcapObjectKeysOnline.onlineAuthentication           = cms.string( DBAuth )
    process.L1TMuonGlobalObjectKeysOnline.onlineAuthentication           = cms.string( DBAuth )
    process.L1TMuonOverlapObjectKeysOnline.onlineAuthentication          = cms.string( DBAuth )
    process.L1TUtmTriggerMenuObjectKeysOnline.onlineAuthentication       = cms.string( DBAuth )


