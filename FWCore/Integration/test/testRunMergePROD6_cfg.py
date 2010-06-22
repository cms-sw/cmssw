import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("EmptySource",
    firstLuminosityBlock = cms.untracked.uint32(2),
    numberEventsInLuminosityBlock = cms.untracked.uint32(3),
    firstEvent = cms.untracked.uint32(2),
    firstRun = cms.untracked.uint32(11),
    numberEventsInRun = cms.untracked.uint32(9)
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('m1')
)

# These are here only for tests of parentage merging
process.m1 = cms.EDProducer("ThingWithMergeProducer")
process.m2 = cms.EDProducer("ThingWithMergeProducer")
process.m3 = cms.EDProducer("ThingWithMergeProducer")

process.tryNoPut = cms.EDProducer("ThingWithMergeProducer",
    noPut = cms.untracked.bool(True)
)

process.makeThingToBeDropped = cms.EDProducer("ThingWithMergeProducer")

process.A = cms.EDProducer("ThingWithMergeProducer")

process.B = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('A')
)

process.C = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('A')
)

process.D = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('B')
)

process.E = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('B', 'C')
)

process.F = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('C')
)

process.G = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('A')
)

process.H = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('G')
)

process.I = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('A')
)

process.J = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('I')
)

process.K = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('I')
)

process.L = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('F')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMerge6.root')
)

process.p1 = cms.Path((process.m1 + process.m2 + process.m3) *
                     process.thingWithMergeProducer *
                     process.tryNoPut *
                     process.makeThingToBeDropped)

process.p2 = cms.Path(process.A *
                      process.B *
                      process.C *
                      process.D *
                      process.E *
                      process.F *
                      process.G *
                      process.H *
                      process.I *
                      process.J *
                      process.K *
                      process.L)

process.e = cms.EndPath(process.out)
