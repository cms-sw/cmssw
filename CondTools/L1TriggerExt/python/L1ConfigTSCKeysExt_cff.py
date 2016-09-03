from L1TriggerConfig.L1TUtmTriggerMenuProducers.L1TUtmTriggerMenuObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TMuonBarrelParamsProducers.L1TMuonBarrelObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TMuonGlobalParamsProducers.L1TMuonGlobalObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TMuonOverlapParamsProducers.L1TMuonOverlapObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TMuonEndcapParamsProducers.L1TMuonEndcapObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TCaloParamsProducers.L1TCaloParamsObjectKeysOnline_cfi import *
from L1TriggerConfig.L1TGlobalPrescalesVetosProducers.L1TGlobalPrescalesVetosObjectKeysOnline_cfi import *

def setTSCKeysDBAuth(process, DBAuth = '.'):
    process.L1TCaloParamsObjectKeysOnline.onlineAuthentication           = cms.string( DBAuth )
    process.L1TGlobalPrescalesVetosObjectKeysOnline.onlineAuthentication = cms.string( DBAuth )
    process.L1TMuonBarrelObjectKeysOnline.onlineAuthentication           = cms.string( DBAuth )
    process.L1TMuonEndcapObjectKeysOnline.onlineAuthentication           = cms.string( DBAuth )
    process.L1TMuonGlobalObjectKeysOnline.onlineAuthentication           = cms.string( DBAuth )
    process.L1TMuonOverlapObjectKeysOnline.onlineAuthentication          = cms.string( DBAuth )
    process.L1TUtmTriggerMenuObjectKeysOnline.onlineAuthentication       = cms.string( DBAuth )


