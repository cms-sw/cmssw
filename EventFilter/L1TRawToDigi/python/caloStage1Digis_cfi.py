import FWCore.ParameterSet.Config as cms

caloStage1Digis = cms.EDProducer(
    "L1TRawToDigi",
    Setup = cms.string("stage1::CaloSetup"),
    FedIds = cms.vint32(1352),
    # Uncomment the following for 74x legacy MC
    # FWOverride = cms.bool(True)
    # FWId = cms.uint32(0xff000000),
)
