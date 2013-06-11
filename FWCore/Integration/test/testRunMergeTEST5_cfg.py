import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST5")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMergeMERGE5.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
        'file:testRunMerge2extra.root'
    )
)

process.test = cms.EDAnalyzer("TestMergeResults",
   testAlias = cms.untracked.bool(True)
)

process.path1 = cms.Path(process.test)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMergeTEST5.root')
)

process.e = cms.EndPath(process.out)
