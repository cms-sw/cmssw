import FWCore.ParameterSet.Config as cms

simGtDigis = cms.EDProducer("L1GlobalTrigger",
    AlgorithmTriggersUnmasked = cms.bool(False),
    AlgorithmTriggersUnprescaled = cms.bool(False),
    AlternativeNrBxBoardDaq = cms.uint32(0),
    AlternativeNrBxBoardEvm = cms.uint32(0),
    BstLengthBytes = cms.int32(-1),
    CastorInputTag = cms.InputTag("castorL1Digis"),
    EmulateBxInEvent = cms.int32(3),
    GctInputTag = cms.InputTag("simGctDigis"),
    GmtInputTag = cms.InputTag("simGmtDigis"),
    ProduceL1GtDaqRecord = cms.bool(True),
    ProduceL1GtEvmRecord = cms.bool(True),
    ProduceL1GtObjectMapRecord = cms.bool(True),
    ReadTechnicalTriggerRecords = cms.bool(True),
    RecordLength = cms.vint32(3, 0),
    TechnicalTriggersInputTags = cms.VInputTag("simBscDigis", "simRpcTechTrigDigis", "simHcalTechTrigDigis", "simCastorTechTrigDigis"),
    TechnicalTriggersUnmasked = cms.bool(False),
    TechnicalTriggersUnprescaled = cms.bool(False),
    TechnicalTriggersVetoUnmasked = cms.bool(False),
    WritePsbL1GtDaqRecord = cms.bool(True)
)
