import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1

process.triggerResultsProducer = cms.EDProducer("TestWriteTriggerResults",
    # Test values below are meaningless. We just make sure when we read
    # we get the same values.
    parameterSetID = cms.string('8b99d66b6c3865c75e460791f721202d'),
    # names should normally be empty. Only extremely old data or
    # has names filled and not empty. If it is not empty, the
    # ParameterSetID is ignored and left default constructed.
    names = cms.vstring(),
    hltStates = cms.vuint32(0, 1, 2, 3),
    moduleIndexes = cms.vuint32(11, 21, 31, 41)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testTriggerResults.root')
)

process.path = cms.Path(process.triggerResultsProducer)
process.endPath = cms.EndPath(process.out)
