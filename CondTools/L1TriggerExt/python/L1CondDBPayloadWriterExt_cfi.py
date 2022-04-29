import FWCore.ParameterSet.Config as cms

L1CondDBPayloadWriterExt = cms.EDAnalyzer(
    "L1CondDBPayloadWriterExt",
    writeL1TriggerKeyExt = cms.bool(True),
    writeConfigData = cms.bool(True),
    overwriteKeys = cms.bool(False),
    logTransactions = cms.bool(False),
    newL1TriggerKeyListExt = cms.bool(False),
    sysWriters = cms.vstring(
        "L1TriggerKeyExtRcd@L1TriggerKeyExt",
        "L1TCaloParamsO2ORcd@CaloParams",
        "L1TGlobalPrescalesVetosFractO2ORcd@L1TGlobalPrescalesVetosFract",
        "L1TMuonBarrelParamsO2ORcd@L1TMuonBarrelParams",
        "L1TMuonEndCapForestO2ORcd@L1TMuonEndCapForest",
        "L1TMuonEndCapParamsO2ORcd@L1TMuonEndCapParams",
        "L1TMuonGlobalParamsO2ORcd@L1TMuonGlobalParams",
        "L1TMuonOverlapFwVersionO2ORcd@L1TMuonOverlapFwVersion",
        "L1TMuonOverlapParamsO2ORcd@L1TMuonOverlapParams",
        "L1TUtmTriggerMenuO2ORcd@L1TUtmTriggerMenu"
    )
)


