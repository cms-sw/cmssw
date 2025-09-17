import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2")

process.options = cms.untracked.PSet(
  fileMode = cms.untracked.string('NOMERGE')
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testDropOnInput1_1.root',
        'file:testDropOnInput1_2.root'
    ),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_prod1_*_*'
    ),
    dropDescendantsOfDroppedBranches = cms.untracked.bool(True)
)

process.prodD = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("prodC"),
    onlyGetOnEvent = cms.untracked.uint32(1)
)

process.prodE = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("prodD"),
    onlyGetOnEvent = cms.untracked.uint32(2)
)

process.K103 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("A101")
)

process.K203 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("K201")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testDropOnInput2.root')
)

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(
    cms.InputTag("prodA"),
    cms.InputTag("prodB"),
    cms.InputTag("prodC")
  ),
  inputTagsNotFound = cms.untracked.VInputTag(
    cms.InputTag("prod1"),
    cms.InputTag("prod2"),
    cms.InputTag("prod3")
  )
)

process.path = cms.Path(process.prodD + process.prodE + process.a1)
process.path4 = cms.Path(process.K103 + process.K203)

process.endpath = cms.EndPath(process.out)
