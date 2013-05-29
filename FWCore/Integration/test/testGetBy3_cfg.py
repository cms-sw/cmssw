import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD3")

process.options = cms.untracked.PSet( allowUnscheduled = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testGetBy2.root'
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGetBy3.root'),
    outputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_intProducerA_*_*'
    )
)

process.intProducer = cms.EDProducer("IntProducer", ivalue = cms.int32(3))

process.intProducerU = cms.EDProducer("IntProducer", ivalue = cms.int32(30))

process.intProducerA = cms.EDProducer("IntProducer", ivalue = cms.int32(200))

process.aliasForInt = cms.EDAlias(
  intProducerA  = cms.VPSet(
    cms.PSet(type = cms.string('edmtestIntProduct')
    )
  )
)

process.nonProducer = cms.EDProducer("NonProducer")

process.intVectorSetProducer = cms.EDProducer("IntVectorSetProducer")

process.intVectorProducer = cms.EDProducer("IntVectorProducer",
  count = cms.int32(9),
  ivalue = cms.int32(31)
)

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducer") ),
  expectedSum = cms.untracked.int32(9)
)

process.a2 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducer", skipCurrentProcess = True) ),
  expectedSum = cms.untracked.int32(6)
)

process.a3 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducer", "", "PROD1") ),
  expectedSum = cms.untracked.int32(3)
)

process.a4 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducer", "", "PROD2") ),
  expectedSum = cms.untracked.int32(6)
)

process.a5 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducer", "", "PROD3") ),
  expectedSum = cms.untracked.int32(9)
)

process.a10 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerU") ),
  expectedSum = cms.untracked.int32(90)
)

process.a20 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerU", skipCurrentProcess = True) ),
  expectedSum = cms.untracked.int32(60)
)

process.a30 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerU", "", "PROD1") ),
  expectedSum = cms.untracked.int32(30)
)

process.a40 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerU", "", "PROD2") ),
  expectedSum = cms.untracked.int32(60)
)

process.a50 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerU", "", "PROD3") ),
  expectedSum = cms.untracked.int32(90)
)

process.a100 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("source", skipCurrentProcess = True) ),
  expectedSum = cms.untracked.int32(12)
)

process.a200 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerA") ),
  expectedSum = cms.untracked.int32(600)
)

process.a300 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("aliasForInt") ),
  expectedSum = cms.untracked.int32(600)
)

process.a400 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("aliasForInt", skipCurrentProcess = True) ),
  expectedSum = cms.untracked.int32(300),
  inputTagsNotFound = cms.untracked.VInputTag(
    cms.InputTag("intProducerA", skipCurrentProcess = True)
  )
)

process.a1000 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("doesNotExist") ),
  expectedSum = cms.untracked.int32(1),
  getByTokenFirst = cms.untracked.bool(True)
)

process.a1001 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("nonProducer") ),
  expectedSum = cms.untracked.int32(1),
  getByTokenFirst = cms.untracked.bool(True)
)

process.a1002 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsView = cms.untracked.VInputTag( cms.InputTag("intVectorSetProducer", "", "PROD3") ),
  expectedSum = cms.untracked.int32(1),
  getByTokenFirst = cms.untracked.bool(True)
)

process.a1003 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsView = cms.untracked.VInputTag( cms.InputTag("intVectorSetProducer") ),
  expectedSum = cms.untracked.int32(1),
  getByTokenFirst = cms.untracked.bool(True)
)

process.a1004 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsView = cms.untracked.VInputTag( cms.InputTag("intVectorProducer") ),
  expectedSum = cms.untracked.int32(93)
)

process.a1005 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  inputTagsView = cms.untracked.VInputTag( cms.InputTag("intVectorProducer", skipCurrentProcess = True) ),
  expectedSum = cms.untracked.int32(63)
)

process.p = cms.Path(process.intProducer * process.a1 * process.a2 * process.a3 * process.a4 * process.a5)

process.p0 = cms.Path(process.a10 * process.a20 * process.a30 * process.a40 * process.a50)

process.p00 = cms.Path(process.a100 * process.a200 * process.a300 * process.a400)

# Cause exception with a module label that does not exist
#process.p1000 = cms.Path(process.a1000)

# Cause exception where product branch exists but
# product was not put into the event
#process.p1001 = cms.Path(process.a1001)

# Cause exception where product request is ambiguous
# and process is specified
#process.p1002 = cms.Path(process.a1002)

# Cause exception where product request is ambiguous
# and process is not specified
#process.p1003 = cms.Path(process.a1003)

process.p1004 = cms.Path(process.a1004)

process.p1005 = cms.Path(process.a1005)

process.e = cms.EndPath(process.out)


