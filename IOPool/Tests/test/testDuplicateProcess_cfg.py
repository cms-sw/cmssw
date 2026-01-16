import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD1")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testGetBy1.root'
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRepeatProcess.root')
)

process.e = cms.EndPath(process.out)
