import FWCore.ParameterSet.Config as cms

process = cms.Process("FOURTH")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:testEventHistory_3.root')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testEventHistory_4.root')
)

process.ep4 = cms.EndPath(process.out)
