import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE5")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#Contents of file
#testRunMerge2extra.root  "PROD" & "EXTRA" [run:1, lumi:1, ev:21-25] [run:2,lumi:1, ev:1-5]

process.source = cms.Source("RNTupleTempSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge2extra.root'
    )
)

process.out = cms.OutputModule("RNTupleTempOutputModule",
    fileName = cms.untracked.string('testRunMergeMERGE5.root'),
    outputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_aliasForThingToBeDropped2_*_*'
    )
)
#Contents of file
#testRunMergeMERGE5.root  "PROD" & "EXTRA" [run:1, lumi:1, ev:21] with "Prod" and "EXTRA" range [run:1, lumi:1, ev:1-5,21-25], excludes Run:2

process.e = cms.EndPath(process.out)
