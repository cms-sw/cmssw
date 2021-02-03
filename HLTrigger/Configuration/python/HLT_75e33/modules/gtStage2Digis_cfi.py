import FWCore.ParameterSet.Config as cms

gtStage2Digis = cms.EDProducer("L1TRawToDigi",
    FedIds = cms.vint32(1404),
    InputLabel = cms.InputTag("rawDataCollector"),
    MinFeds = cms.uint32(1),
    Setup = cms.string('stage2::GTSetup')
)
