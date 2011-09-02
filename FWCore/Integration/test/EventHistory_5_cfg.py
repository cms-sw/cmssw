import FWCore.ParameterSet.Config as cms

process = cms.Process("FIFTH")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:testEventHistory_4.root')
)

process.out = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('f55:SECOND','f75:SECOND')
    ),
    fileName = cms.untracked.string('testEventHistory_5.root')
)

process.ep4 = cms.EndPath(process.out)
