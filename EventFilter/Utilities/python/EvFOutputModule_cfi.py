import FWCore.ParameterSet.Config as cms

evfOutputModule = cms.OutputModule("EvFOutputModule",
    fileName = cms.untracked.string(""),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(),
        outputCommands = cms.untracked.vstring()
    )
)
