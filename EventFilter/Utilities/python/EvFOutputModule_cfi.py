import FWCore.ParameterSet.Config as cms

process.EvFOutputModule = cms.Service("EvFOutputModule",
    fileName = cms.untracked.string(""),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(),
        outputCommands = cms.untracked.vstring()
    )
)

