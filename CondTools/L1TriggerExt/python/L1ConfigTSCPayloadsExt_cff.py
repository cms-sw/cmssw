from L1TriggerConfig.L1TUtmTriggerMenuProducers.L1TUtmTriggerMenuOnline_cfi import *
from L1TriggerConfig.L1TMuonBarrelParamsProducers.L1TMuonBarrelParamsOnline_cfi import *
from L1TriggerConfig.L1TMuonGlobalParamsProducers.L1TMuonGlobalParamsOnline_cfi import *
from L1TriggerConfig.L1TMuonOverlapParamsProducers.L1TMuonOverlapParamsOnline_cfi import *
from L1TriggerConfig.L1TMuonEndcapParamsProducers.L1TMuonEndcapParamsOnline_cfi import *
from L1TriggerConfig.L1TCaloParamsProducers.L1TCaloParamsOnline_cfi import *
from L1TriggerConfig.L1TGlobalPrescalesVetosProducers.L1TGlobalPrescalesVetosOnline_cfi import *

def setTSCPayloadsDBAuth(process, DBAuth = '.'):
    process.L1TCaloParamsOnlineProd.onlineAuthentication           = cms.string( DBAuth )
    process.L1TGlobalPrescalesVetosOnlineProd.onlineAuthentication = cms.string( DBAuth )
    process.L1TMuonBarrelParamsOnlineProd.onlineAuthentication     = cms.string( DBAuth )
    process.L1TMuonEndcapParamsOnlineProd.onlineAuthentication     = cms.string( DBAuth )
    process.L1TMuonGlobalParamsOnlineProd.onlineAuthentication     = cms.string( DBAuth )
    process.L1TMuonOverlapParamsOnlineProd.onlineAuthentication    = cms.string( DBAuth )
    process.L1TUtmTriggerMenuOnlineProd.onlineAuthentication       = cms.string( DBAuth )

