import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST3")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testDropOnInput2.root'
    ),
    inputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_prodA_*_*',
        'drop *_prodD_*_*',
        'drop *_prodF_*_*',
        'drop *_A101_*_*',
        'drop *_K201_*_*'
    ),
    dropDescendantsOfDroppedBranches = cms.untracked.bool(True)
)

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(
      cms.InputTag("prodC"),
      cms.InputTag("prodE"),
      cms.InputTag("prodG"),
      cms.InputTag("K100"),
      cms.InputTag("K200")      
  ),
  inputTagsNotFound = cms.untracked.VInputTag(
      cms.InputTag("prodA"),
      cms.InputTag("prodD"),
      cms.InputTag("prodB"),
      cms.InputTag("prodF"),
      cms.InputTag("NK101"),
      cms.InputTag("A101"),
      cms.InputTag("K102"),
      cms.InputTag("K103"),
      cms.InputTag("K104"),
      cms.InputTag("K201"),
      cms.InputTag("A201"),
      cms.InputTag("K202"),
      cms.InputTag("K203"),
      cms.InputTag("K204")
  )
)

process.path = cms.Path(process.a1)
