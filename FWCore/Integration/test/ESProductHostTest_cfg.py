# Test the ESProductHost class

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# start a new run every event
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1)
)

# control when the event setup records get updated
process.emptyESSourceB = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordB"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8),
    iovIsRunNotTime = cms.bool(True)
)

process.emptyESSourceC = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordC"),
    firstValid = cms.vuint32(1,3,5),
    iovIsRunNotTime = cms.bool(True)
)

# This ESProducer uses the ESProductHost and
# also tests it.
process.esTestProducerBUsingHost = cms.ESProducer("ESTestProducerBUsingHost")

# Read the values produced by the ESProducer and test
# that they match expectations
process.esTestAnalyzerB = cms.EDAnalyzer("ESTestAnalyzerB",
    runsToGetDataFor = cms.vint32(1,2,3,4,5,6,7,8,9,10),
    expectedValues = cms.untracked.vint32(2,3,5,6,8,9,10,11,11,11)
)

process.p = cms.Path(process.esTestAnalyzerB)
