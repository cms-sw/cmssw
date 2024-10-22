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
    firstLuminosityBlock = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(5),
    firstEvent = cms.untracked.uint32(21),
    firstRun = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(5)
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('m2')
)

# These are here only for tests of parentage merging
process.m1 = cms.EDProducer("ThingWithMergeProducer")
process.m2 = cms.EDProducer("ThingWithMergeProducer")
process.m3 = cms.EDProducer("ThingWithMergeProducer")

process.tryNoPut = cms.EDProducer("ThingWithMergeProducer",
    noPut = cms.untracked.bool(True)
)

# This one tests products dropped and then restored by secondary file
# input
process.makeThingToBeDropped = cms.EDProducer("ThingWithMergeProducer")

process.makeThingToBeDropped2 = cms.EDProducer("ThingWithMergeProducer")

process.aliasForThingToBeDropped2 = cms.EDAlias(
    makeThingToBeDropped2  = cms.VPSet(
      cms.PSet(type = cms.string('edmtestThing'),
               fromProductInstance = cms.string('event'),
               toProductInstance = cms.string('instance2')),
      cms.PSet(type = cms.string('edmtestThing'),
               fromProductInstance = cms.string('endLumi'),
               toProductInstance = cms.string('endLumi2')),
      cms.PSet(type = cms.string('edmtestThing'),
               fromProductInstance = cms.string('endRun'),
               toProductInstance = cms.string('endRun2'))
    )
)

# This product will be produced in configuration PROD1 and PROD5
# In PROD2 it will be produced and dropped and there will be another
# product whose provenance includes it as a parent. In PROD3 it will
# be produced and dropped and there will not be a product that includes
# it as a parent. In PROD4 it will never be produced at all.
process.makeThingToBeDropped1 = cms.EDProducer("ThingWithMergeProducer")
process.dependsOnThingToBeDropped1 = cms.EDProducer("ThingWithMergeProducer",
    labelsToGet = cms.untracked.vstring('makeThingToBeDropped1')
)

process.test = cms.EDAnalyzer("TestMergeResults",

    #   Check to see that the value we read matches what we know
    #   was written. Expected values listed below come in sets of three
    #      value expected in Thing
    #      value expected in ThingWithMerge
    #      value expected in ThingWithIsEqual
    #   Each set of 3 is tested at endRun for the expected
    #   run values or at endLuminosityBlock for the expected
    #   lumi values. And then the next set of three values
    #   is tested at the next endRun or endLuminosityBlock.
    #   When the sequence of parameter values is exhausted it stops checking
    #   0's are just placeholders, if the value is a "0" the check is not made.
    expectedBeginRunProd = cms.untracked.vint32(
        10001,   10002,  10003,
        10001,   10002,  10003
    ),

    expectedEndRunProd = cms.untracked.vint32(
        100001, 100002, 100003,
        100001, 100002, 100003
    ),

    expectedBeginLumiProd = cms.untracked.vint32(
        101,       102,    103,
        101,       102,    103
    ),

    expectedEndLumiProd = cms.untracked.vint32(
        1001,     1002,   1003,
        1001,     1002,   1003
    ),

    verbose = cms.untracked.bool(False),

    expectedParents = cms.untracked.vstring(
        'm2', 'm2', 'm2', 'm2', 'm2',
        'm2', 'm2', 'm2', 'm2', 'm2'),
    testAlias = cms.untracked.bool(True)
)

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
    fileName = cms.untracked.string('testRunMerge2.root'),
    outputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_makeThingToBeDropped1_*_*',
        'drop *_makeThingToBeDropped2_*_*'
    )
)

process.p1 = cms.Path((process.m1 + process.m2 + process.m3) *
                     process.thingWithMergeProducer *
                     process.makeThingToBeDropped2 *
                     process.test *
                     process.tryNoPut *
                     process.makeThingToBeDropped *
                     process.makeThingToBeDropped1 *
                     process.dependsOnThingToBeDropped1)

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
