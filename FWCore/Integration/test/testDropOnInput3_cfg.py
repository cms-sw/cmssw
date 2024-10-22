import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testDropOnInput1_1.root',
        'file:testDropOnInput1_2.root'
    ),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_prod1_*_*'
    ),
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False)
)

process.prodD = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("prodC"),
    onlyGetOnEvent = cms.untracked.uint32(1)
)

process.prodE = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("prodD"),
    onlyGetOnEvent = cms.untracked.uint32(2)
)

process.K1 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag()
)

process.K2 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag()
)

process.NK1 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("K1", "K2")
)

process.NK2 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("NK1")
)

process.NK3 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("NK2")
)

process.K3 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag()
)

process.K4 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag()
)

process.NK4 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("K3", "K4")
)

process.NK5 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("NK4")
)

process.K5 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("NK5")
)

process.NK6 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("K4")
)

process.NK7 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("NK6")
)

process.NK8 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("NK6")
)

process.K103 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("A101")
)

process.K203 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("K201")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testDropOnInput3.root'),
    outputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_NK*_*_*'
    )
)

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(
      cms.InputTag("prod3"),
      cms.InputTag("prodA"),
      cms.InputTag("prodB"),
      cms.InputTag("prodC"),
      cms.InputTag("prodD"),
      cms.InputTag("prodE")
  ),
  inputTagsNotFound = cms.untracked.VInputTag(
      cms.InputTag("prod1"),
      cms.InputTag("prod2")
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

process.path1 = cms.Path(process.prodD + process.prodE + process.a1)
process.path2 = cms.Path(process.K1 + process.K2 + process.NK1 + process.NK2 + process.NK3)
process.path3 = cms.Path(process.K3 + process.K4 + process.NK4 + process.NK5 +
                         process.K5 + process.NK6 + process.NK7 +
                         process.NK8)
process.path4 = cms.Path(process.K103 + process.K203)

process.endpath = cms.EndPath(process.out * process.test1 * process.test2 *
                              process.test3 * process.test4 * process.test5)
