import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE5")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge2extra.root'
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMergeMERGE5.root'),
    outputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_aliasForThingToBeDropped2_*_*'
    )
)

process.e = cms.EndPath(process.out)
