import FWCore.ParameterSet.Config as cms

gtStage2Digis = cms.EDProducer(
    "L1TRawToDigi",
    InputLabel = cms.InputTag("rawDataCollector"),
    Setup           = cms.string("stage2::GTSetup"),
    FedIds          = cms.vint32( 1404 ),
)
