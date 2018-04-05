import FWCore.ParameterSet.Config as cms

gtStage2Digis = cms.EDProducer(
    "L1TRawToDigi",
    Setup           = cms.string("stage2::GTSetup"),
    FedIds          = cms.vint32( 1404 ),
)
