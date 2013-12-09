import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True),
    Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow,
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)
process.source = cms.Source("EmptySource",
    timeBetweenEvents = cms.untracked.uint64(10),
    firstTime = cms.untracked.uint64(1000000)
)

process.Tracer = cms.Service('Tracer',
                             dumpContextForLabels = cms.untracked.vstring('one')
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations   = cms.untracked.vstring('cout',
                                           'cerr'
    ),
    categories = cms.untracked.vstring(
        'Tracer'
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet (
            limit = cms.untracked.int32(0)
        ),
        Tracer = cms.untracked.PSet(
            limit=cms.untracked.int32(100000000)
        )
    )
)

process.one = cms.EDProducer("IntProducer",
    ivalue = cms.int32(1)
)

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
