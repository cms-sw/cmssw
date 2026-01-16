import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST3")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testDropOnInput2001.root'
    ),
    inputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_prodA_*_*',
        'drop *_prodD_*_*',
        'drop *_prodF_*_*'
    ),
    dropDescendantsOfDroppedBranches = cms.untracked.bool(True)
)

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(
      cms.InputTag("prodC"),
      cms.InputTag("prodB")
  ),
  inputTagsNotFound = cms.untracked.VInputTag(
      cms.InputTag("prodA"),
      cms.InputTag("prodD"),
      cms.InputTag("prodE"),
      cms.InputTag("prodF"),
      cms.InputTag("prodG")
  )
)

process.test1 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("A101"),
                               expectedAncestors = cms.vstring("K100")
)

process.test2 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("K100"),
                               expectedAncestors = cms.vstring()
)

process.test3 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("K102"),
                               expectedAncestors = cms.vstring("K100", "NK101")
)

process.test4 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("K103"),
                               expectedAncestors = cms.vstring("K100", "NK101")
)

process.test5 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("K104"),
                               expectedAncestors = cms.vstring("K100", "NK101")
)

process.path = cms.Path(process.a1 *
                        process.test1 *
                        process.test2 *
                        process.test3 *
                        process.test4 *
                        process.test5
)
