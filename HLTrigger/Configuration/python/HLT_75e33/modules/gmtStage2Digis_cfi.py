import FWCore.ParameterSet.Config as cms

gmtStage2Digis = cms.EDProducer("L1TRawToDigi",
    FedIds = cms.vint32(1402),
    InputLabel = cms.InputTag("rawDataCollector"),
    MinFeds = cms.uint32(1),
    Setup = cms.string('stage2::GMTSetup')
)
