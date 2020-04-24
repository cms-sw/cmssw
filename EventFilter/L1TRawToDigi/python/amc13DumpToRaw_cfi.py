import FWCore.ParameterSet.Config as cms

amc13DumpToRaw = cms.EDProducer(
    "AMC13DumpToRaw",

    filename         = cms.untracked.string("data.txt"),
    fedId            = cms.untracked.int32(1352)

)
