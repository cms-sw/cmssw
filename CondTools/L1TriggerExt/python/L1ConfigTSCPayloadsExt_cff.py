from L1TriggerConfig.L1TConfigProducers.L1TUtmTriggerMenuOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonBarrelParamsOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonGlobalParamsOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonOverlapParamsOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonEndcapParamsOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TMuonEndcapForestOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TCaloParamsOnline_cfi import *
from L1TriggerConfig.L1TConfigProducers.L1TGlobalPrescalesVetosOnline_cfi import *

def setTSCPayloadsDB(process, DBConnect, DBAuth, protoDBConnect, protoDBAuth):

    process.L1TCaloParamsOnlineProd.onlineDB           = cms.string( DBConnect )
    process.L1TGlobalPrescalesVetosOnlineProd.onlineDB = cms.string( DBConnect )
    process.L1TMuonBarrelParamsOnlineProd.onlineDB     = cms.string( DBConnect )
    process.L1TMuonEndcapParamsOnlineProd.onlineDB     = cms.string( DBConnect )
    process.L1TMuonEndcapForestOnlineProd.onlineDB     = cms.string( DBConnect )
    process.L1TMuonGlobalParamsOnlineProd.onlineDB     = cms.string( DBConnect )
    process.L1TMuonOverlapParamsOnlineProd.onlineDB    = cms.string( DBConnect )
    process.L1TUtmTriggerMenuOnlineProd.onlineDB       = cms.string( DBConnect )

    process.L1TCaloParamsOnlineProd.onlineAuthentication           = cms.string( DBAuth )
    process.L1TGlobalPrescalesVetosOnlineProd.onlineAuthentication = cms.string( DBAuth )
    process.L1TMuonBarrelParamsOnlineProd.onlineAuthentication     = cms.string( DBAuth )
    process.L1TMuonEndcapParamsOnlineProd.onlineAuthentication     = cms.string( DBAuth )
    process.L1TMuonEndcapForestOnlineProd.onlineAuthentication     = cms.string( DBAuth )
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
