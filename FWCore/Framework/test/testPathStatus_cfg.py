import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(7)
)

# The TestGetPathStatus analyzer tests that the PathStatus
# and EndPathStatus can be read from the event and that
# the PathStatus product contains the correct values.

process.a1 = cms.EDAnalyzer("TestGetPathStatus",
    pathStatusTag = cms.InputTag("path1"),
    endPathStatusTag = cms.InputTag("endpath2"),
    # The index into these two vectors is the EventID.
    # The EventID starts at 1 so the first element is ignored.
    expectedStates = cms.vint32(0,2,2,2,2,2,1,2),
    expectedIndexes = cms.vuint32(0,1,2,1,2,1,2,1)
)

# Same test with an empty path and empty endpath

process.a2 = cms.EDAnalyzer("TestGetPathStatus",
    pathStatusTag = cms.InputTag("path2"),
    endPathStatusTag = cms.InputTag("endpath3"),
    expectedStates = cms.vint32(0,1,1,1,1,1,1,1),
    expectedIndexes = cms.vuint32(0,0,0,0,0,0,0,0)
)

process.f1 = cms.EDFilter("ModuloEventIDFilter",
    modulo = cms.uint32(2),
    offset = cms.uint32(0)
)

process.f2 = cms.EDFilter("ModuloEventIDFilter",
    modulo = cms.uint32(3),
    offset = cms.uint32(0)
)

process.prod1 = cms.EDProducer("IntProducer", ivalue = cms.int32(1))

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testPathStatus.root')
)

process.path1 = cms.Path(process.prod1 * process.f1 * process.f2)
process.path2 = cms.Path()

process.endpath1 = cms.EndPath(process.a1 * process.a2)
process.endpath2 = cms.EndPath(process.out)
process.endpath3 = cms.EndPath()
