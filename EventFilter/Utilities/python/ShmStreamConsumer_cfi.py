import FWCore.ParameterSet.Config as cms

process.ShmStreamConsumer = cms.Service("ShmStreamConsumer",
    fileName = cms.untracked.string(""),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(),
        outputCommands = cms.untracked.vstring()
    )
)

