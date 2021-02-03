import FWCore.ParameterSet.Config as cms

gtDigis = cms.EDProducer("L1GlobalTriggerRawToDigi",
    ActiveBoardsMask = cms.uint32(65535),
    DaqGtFedId = cms.untracked.int32(813),
    DaqGtInputTag = cms.InputTag("rawDataCollector"),
    UnpackBxInEvent = cms.int32(-1),
    Verbosity = cms.untracked.int32(0),
    mightGet = cms.optional.untracked.vstring
)
