import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST5")

#Contents of file
#testRunMergeMERGE5.root  "PROD" & "EXTRA" [run:1, lumi:1, ev:21] with "Prod" and "EXTRA" range [run:1, lumi:1, ev:1-5,21-25], excludes Run:2

#Contents of file
#testRunMerge2extra.root  "PROD" & "EXTRA" [run:1, lumi:1, ev:21-25] [run:2,lumi:1, ev:1-5]

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
