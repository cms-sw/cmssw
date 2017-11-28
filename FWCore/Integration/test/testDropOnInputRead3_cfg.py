import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST3")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testDropOnInput3.root'
    ),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_prodA_*_*',
        'drop *_K100_*_*',
        'drop *_K200_*_*'
    ),
    dropDescendantsOfDroppedBranches = cms.untracked.bool(True)
)

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(
    cms.InputTag("prod3")
  ),

  inputTagsNotFound = cms.untracked.VInputTag(
    cms.InputTag("prod1"),
    cms.InputTag("prod2"),
    cms.InputTag("prodA"),
    cms.InputTag("prodB"),
    cms.InputTag("prodC"),
    cms.InputTag("prodD"),
    cms.InputTag("prodE"),
    cms.InputTag("K100"),
    cms.InputTag("NK101"),
    cms.InputTag("A101"),
    cms.InputTag("K102"),
    cms.InputTag("K103"),
    cms.InputTag("K104"),
    cms.InputTag("K200"),
    cms.InputTag("K201"),
    cms.InputTag("A201"),
    cms.InputTag("K202"),
    cms.InputTag("K203"),
    cms.InputTag("K204")
  )
)

process.test1 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("K5"),
                               expectedAncestors = cms.vstring("K3", "K4", "NK4", "NK5")
)

process.test2 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("prod3"),
                               # prod1 gets converted to an empty string in the TestParentage module
                               # because drop on input removes it from the ProductRegistry
                               # completely, but its BranchID still appears in the Parentage.
                               expectedAncestors = cms.vstring("prod2", ""),
                               callGetProvenance = cms.untracked.bool(False)
)

process.path = cms.Path(process.a1 * process.test1 * process.test2)
