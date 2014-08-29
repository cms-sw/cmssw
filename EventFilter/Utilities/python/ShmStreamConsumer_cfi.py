import FWCore.ParameterSet.Config as cms

shmStreamConsumer = cms.OutputModule("ShmStreamConsumer",
    fileName = cms.untracked.string(""),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(),
        outputCommands = cms.untracked.vstring()
    )
)
