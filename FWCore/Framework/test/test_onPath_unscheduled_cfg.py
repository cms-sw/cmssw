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

process.Tracer = cms.Service("Tracer")

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

process.p = cms.Path(process.one*process.getOne+process.two*process.getTwo)


