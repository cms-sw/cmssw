import FWCore.ParameterSet.Config as cms

caloStage2Digis = cms.EDProducer("L1TRawToDigi",
    FWId = cms.uint32(0),
    FWOverride = cms.bool(False),
    FedIds = cms.vint32(1360, 1366),
    InputLabel = cms.InputTag("rawDataCollector"),
    MinFeds = cms.uint32(1),
    Setup = cms.string('stage2::CaloSetup'),
    TMTCheck = cms.bool(True)
)
