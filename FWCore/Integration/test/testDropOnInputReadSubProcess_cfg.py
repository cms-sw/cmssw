import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST3")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testDropOnInputSubProcess.root'
    ),
    inputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_prod1_*_*',
        'drop *_K100_*_*',
        'drop *_K200_*_*'
    ),
    dropDescendantsOfDroppedBranches = cms.untracked.bool(True)
)

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),

  inputTagsNotFound = cms.untracked.VInputTag(
    cms.InputTag("prod1"),
    cms.InputTag("prod2"),
    cms.InputTag("prod3"),
    cms.InputTag("K100"),
    cms.InputTag("NK101"),
    cms.InputTag("A101"),
    cms.InputTag("K102"),
    cms.InputTag("K103"),
    cms.InputTag("K104"),
    cms.InputTag("K105"),
    cms.InputTag("K200"),
    cms.InputTag("K201"),
    cms.InputTag("A201"),
    cms.InputTag("K202"),
    cms.InputTag("K203"),
    cms.InputTag("K204"),
    cms.InputTag("K205")
  )
)

process.path = cms.Path(process.a1)
