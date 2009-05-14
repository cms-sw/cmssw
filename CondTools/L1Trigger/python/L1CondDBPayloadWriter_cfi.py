import FWCore.ParameterSet.Config as cms

L1CondDBPayloadWriter = cms.EDFilter("L1CondDBPayloadWriter",
    offlineDB = cms.string('sqlite_file:l1config.db'),
    L1TriggerKeyListTag = cms.string('L1TriggerKeyList_IDEAL'),
    offlineAuthentication = cms.string(''),
    writeL1TriggerKey = cms.bool(True),
    writeConfigData = cms.bool(True)
)


