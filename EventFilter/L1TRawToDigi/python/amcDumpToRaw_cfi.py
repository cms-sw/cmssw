import FWCore.ParameterSet.Config as cms

amcDumpToRaw = cms.EDProducer(
    "AMCDumpToRaw",

    filename         = cms.untracked.string("data_nohdr.txt"),
    fedId            = cms.untracked.int32(1352),
    iAmc             = cms.untracked.int32(1),
    boardId          = cms.untracked.int32(4109),
    eventType        = cms.untracked.int32(238),
    fwVersion        = cms.untracked.int32(255),
    lenSlinkHeader   = cms.untracked.int32(8),
    lenSlinkTrailer  = cms.untracked.int32(8)

)
