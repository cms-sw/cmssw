import FWCore.ParameterSet.Config as cms

# cfi uGT emulator

simGtStage2Digis = cms.EDProducer("l1t::GtProducer",
    #TechnicalTriggersUnprescaled = cms.bool(False),
    ProduceL1GtObjectMapRecord = cms.bool(True),
    AlgorithmTriggersUnmasked = cms.bool(False),
    EmulateBxInEvent = cms.int32(1),
    L1DataBxInEvent = cms.int32(5),
    AlgorithmTriggersUnprescaled = cms.bool(False),
    ProduceL1GtDaqRecord = cms.bool(True),
    GmtInputTag = cms.InputTag("gtInput"),
    caloInputTag = cms.InputTag("gtInput"),
    AlternativeNrBxBoardDaq = cms.uint32(0),
    #WritePsbL1GtDaqRecord = cms.bool(True),
    TriggerMenuLuminosity = cms.string('startup'),
    PrescaleCSVFile = cms.string('prescale_L1TGlobal.csv'),
    PrescaleSet = cms.uint32(1),
    BstLengthBytes = cms.int32(-1),
    Verbosity = cms.untracked.int32(0)
)

