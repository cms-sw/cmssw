import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(False),
    Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)
process.source = cms.Source("EmptySource",
    timeBetweenEvents = cms.untracked.uint64(10),
    firstTime = cms.untracked.uint64(1000000)
)

process.Tracer = cms.Service("Tracer")

process.one = cms.EDProducer("IntProducer",
    ivalue = cms.int32(1)
)

process.two = cms.EDProducer("IntProducer",
    ivalue = cms.int32(2)
)

process.getOne = cms.EDAnalyzer("IntTestAnalyzer",
    valueMustMatch = cms.untracked.int32(1),
    moduleLabel = cms.untracked.string('one')
)

process.getTwo = cms.EDAnalyzer("IntTestAnalyzer",
    valueMustMatch = cms.untracked.int32(2),
    moduleLabel = cms.untracked.string('two')
)

process.p = cms.Path(process.getOne+process.getTwo)
