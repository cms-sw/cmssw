import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True),
    Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)
process.source = cms.Source("EmptySource",
    timeBetweenEvents = cms.untracked.uint32(10),
    firstTime = cms.untracked.uint32(1000000)
)

process.Tracer = cms.Service("Tracer")

process.result1 = cms.EDProducer("AddIntsProducer",
    labels = cms.vstring('one')
)

process.result2 = cms.EDProducer("AddIntsProducer",
    labels = cms.vstring('result1', 
        'one')
)

process.result4 = cms.EDProducer("AddIntsProducer",
    labels = cms.vstring('result2', 
        'result2')
)

process.get = cms.EDAnalyzer("IntTestAnalyzer",
    valueMustMatch = cms.untracked.int32(4),
    moduleLabel = cms.untracked.string('result4')
)

process.p = cms.Path(process.get)
