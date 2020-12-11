import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
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
    cerr = cms.untracked.PSet(
      enableStatistics = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
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
    labels = cms.VInputTag('one')
)

process.result2 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag('result1',
        'one')
)

process.result4 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag('result2',
        'result2')
)

process.get = cms.EDAnalyzer("IntTestAnalyzer",
    valueMustMatch = cms.untracked.int32(4),
    moduleLabel = cms.untracked.InputTag('result4')
)

process.t = cms.Task(process.one, process.result1, process.result2, process.result4)

process.p = cms.Path(process.get, process.t)
