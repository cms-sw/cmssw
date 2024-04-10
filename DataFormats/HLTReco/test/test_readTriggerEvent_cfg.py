import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+sys.argv[1]))
process.maxEvents.input = 1

process.testReadTriggerEvent = cms.EDAnalyzer("TestReadTriggerEvent",
    expectedUsedProcessName = cms.string("testName"),
    expectedCollectionTags = cms.vstring('moduleA', 'moduleB', 'moduleC'),
    expectedCollectionKeys = cms.vuint32(11, 21, 31),
    expectedIds = cms.vint32(1, 3, 5),
    # I stick to values exactly convertable from double to float
    # to avoid potential rounding issues in the test.
    # (configuration only supports double not float and
    # the data format holds floats)
    expectedPts = cms.vdouble(11.0, 21.0, 31.0),
    expectedEtas = cms.vdouble(101.0, 102.0, 103.0),
    expectedPhis = cms.vdouble(201.0, 202.0, 203.0),
    expectedMasses = cms.vdouble(301.0, 302.0, 303.0),
    expectedFilterTags = cms.vstring('moduleAA', 'moduleBB'),
    expectedElementsPerVector = cms.uint32(2),
    expectedFilterIds = cms.vint32(1001, 1002, 1003, 1004),
    expectedFilterKeys = cms.vuint32(2001, 2002, 2003, 2004),
    triggerEventTag = cms.InputTag("triggerEventProducer", "", "PROD")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testTriggerEvent2.root')
)

process.path = cms.Path(process.testReadTriggerEvent)

process.endPath = cms.EndPath(process.out)
