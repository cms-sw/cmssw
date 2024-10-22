import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(3))

process.options = cms.untracked.PSet(
    canDeleteEarly = cms.untracked.vstring(
        "edmtestDeleteEarly_maker1__TEST",
        "edmtestDeleteEarly_maker2__TEST",
        "edmtestDeleteEarly_maker3__TEST"
    )
)


process.maker1 = cms.EDProducer("DeleteEarlyProducer")
process.maker2 = cms.EDProducer("DeleteEarlyProducer")
process.maker3 = cms.EDProducer("DeleteEarlyProducer") # this module should get deleted

# These 3 modules should get deleted
process.otherProducer1 = cms.EDProducer("IntProducer",
    ivalue = cms.int32(1)
)
process.otherProducer2 = cms.EDProducer("IntProducer",
    ivalue = cms.int32(2)
)
process.otherProducer3 = cms.EDProducer("IntProducer",
    ivalue = cms.int32(3)
)

process.reader1 = cms.EDAnalyzer("DeleteEarlyReader",
    tag = cms.untracked.InputTag("maker1")
)
process.reader2 = cms.EDAnalyzer("DeleteEarlyReader",
    tag = cms.untracked.InputTag("maker2")
)

process.tester = cms.EDAnalyzer("DeleteEarlyCheckDeleteAnalyzer",
    expectedValues = cms.untracked.vuint32(4,8,12)
)

process.t = cms.Task(
    process.maker1,
    process.maker2,
    process.maker3,
    process.otherProducer1,
    process.otherProducer2,
    process.otherProducer3,
)
process.p = cms.Path(process.reader1+process.reader2+process.tester, process.t)
