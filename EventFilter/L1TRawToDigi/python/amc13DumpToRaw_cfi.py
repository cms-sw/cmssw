import FWCore.ParameterSet.Config as cms

amc13DumpToRaw = cms.EDProducer(
    "AMC13DumpToRaw",

    filename         = cms.untracked.string("data.txt"),
    fedId            = cms.untracked.int32(1352)

)
# foo bar baz
# ZCkv4pkV63N5H
# IfNnBUdA0nXnD
