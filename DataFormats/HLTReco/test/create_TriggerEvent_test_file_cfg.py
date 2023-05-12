import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1

process.triggerEventProducer = cms.EDProducer("TestWriteTriggerEvent",
    # Test values below are meaningless. We just make sure when we read
    # we get the same values.
    usedProcessName = cms.string("testName"),
    collectionTags = cms.vstring('moduleA', 'moduleB', 'moduleC'),
    collectionKeys = cms.vuint32(11, 21, 31),
    ids = cms.vint32(1, 3, 5),
    # I stick to values exactly convertable to float
    # to avoid potential rounding issues in the test.
    pts = cms.vdouble(11.0, 21.0, 31.0),
    etas = cms.vdouble(101.0, 102.0, 103.0),
    phis = cms.vdouble(201.0, 202.0, 203.0),
    masses = cms.vdouble(301.0, 302.0, 303.0),
    filterTags = cms.vstring('moduleAA', 'moduleBB'),
    elementsPerVector = cms.uint32(2),
    filterIds = cms.vint32(1001, 1002, 1003, 1004),
    filterKeys = cms.vuint32(2001, 2002, 2003, 2004)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testTriggerEvent.root')
)

process.path = cms.Path(process.triggerEventProducer)
process.endPath = cms.EndPath(process.out)
