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
    numberEventsInLuminosityBlock = cms.untracked.uint32(100),
    firstEvent = cms.untracked.uint32(11),
    firstRun = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(100)
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

# This one tests products dropped and then restored by secondary file
# input
process.makeThingToBeDropped = cms.EDProducer("ThingWithMergeProducer")

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

    verbose = cms.untracked.bool(False),

    expectedParents = cms.untracked.vstring(
        'm1', 'm1', 'm1', 'm1', 'm1',
        'm1', 'm1', 'm1', 'm1', 'm1')
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

process.testGetterOfProducts = cms.EDFilter("TestGetterOfProducts",
    processName = cms.string('PROD'),
    expectedInputTagLabels = cms.vstring('A','B','C','D','E','F',
                                         'G', 'H', 'I', 'J', 'K', 'L',
                                        'dependsOnThingToBeDropped1',
                                        'm1', 'm2', 'm3',
                                        'makeThingToBeDropped',
                                        'makeThingToBeDropped1',
                                        'thingWithMergeProducer',
                                        'tryNoPut' ),
    expectedLabelsAfterGet = cms.vstring('A','B','C','D','E','F',
                                         'G', 'H', 'I', 'J', 'K', 'L',
                                         'dependsOnThingToBeDropped1',
                                         'm1', 'm2', 'm3',
                                         'makeThingToBeDropped',
                                         'makeThingToBeDropped1',
                                         'thingWithMergeProducer'
                                         
    )
)

process.testGetterOfProductsA = cms.EDAnalyzer("TestGetterOfProductsA",
    processName = cms.string('PROD'),
    branchType = cms.int32(0),
    expectedInputTagLabels = cms.vstring('A','B','C','D','E','F',
                                         'G', 'H', 'I', 'J', 'K', 'L',
                                        'dependsOnThingToBeDropped1',
                                        'm1', 'm2', 'm3',
                                        'makeThingToBeDropped',
                                        'makeThingToBeDropped1',
                                        'thingWithMergeProducer',
                                        'tryNoPut'
    ),
    expectedLabelsAfterGet = cms.vstring('A','B','C','D','E','F',
                                         'G', 'H', 'I', 'J', 'K', 'L',
                                         'dependsOnThingToBeDropped1',
                                         'm1', 'm2', 'm3',
                                         'makeThingToBeDropped',
                                         'makeThingToBeDropped1',
                                         'thingWithMergeProducer'
    ),
    expectedNumberOfThingsWithLabelA = cms.uint32(1)
)

process.testGetterOfProductsARun = cms.EDAnalyzer("TestGetterOfProductsA",
    processName = cms.string('PROD'),
    branchType = cms.int32(2),
    expectedInputTagLabels = cms.vstring('A','A','B','B','C','C','D','D','E','E','F','F',
                                         'G', 'G', 'H', 'H', 'I', 'I', 'J', 'J', 'K', 'K', 'L', 'L',
                                        'dependsOnThingToBeDropped1','dependsOnThingToBeDropped1',
                                        'm1', 'm1', 'm2', 'm2', 'm3','m3',
                                        'makeThingToBeDropped','makeThingToBeDropped',
                                        'makeThingToBeDropped1','makeThingToBeDropped1',
                                        'thingWithMergeProducer','thingWithMergeProducer',
                                        'tryNoPut','tryNoPut'
    ),
    expectedLabelsAfterGet = cms.vstring('A','A','B','B','C','C','D','D','E','E','F','F',
                                         'G', 'G', 'H', 'H', 'I', 'I', 'J', 'J', 'K', 'K', 'L', 'L',
                                         'dependsOnThingToBeDropped1', 'dependsOnThingToBeDropped1',
                                         'm1', 'm1', 'm2', 'm2', 'm3', 'm3',
                                         'makeThingToBeDropped', 'makeThingToBeDropped',
                                         'makeThingToBeDropped1', 'makeThingToBeDropped1',
                                         'thingWithMergeProducer', 'thingWithMergeProducer'
    ),
    expectedNumberOfThingsWithLabelA = cms.uint32(2)
)

process.testGetterOfProductsALumi = cms.EDAnalyzer("TestGetterOfProductsA",
    processName = cms.string('PROD'),
    branchType = cms.int32(1),
    expectedInputTagLabels = cms.vstring('A','A','B','B','C','C','D','D','E','E','F','F',
                                         'G', 'G', 'H', 'H', 'I', 'I', 'J', 'J', 'K', 'K', 'L', 'L',
                                        'dependsOnThingToBeDropped1','dependsOnThingToBeDropped1',
                                        'm1', 'm1', 'm2', 'm2', 'm3','m3',
                                        'makeThingToBeDropped','makeThingToBeDropped',
                                        'makeThingToBeDropped1','makeThingToBeDropped1',
                                        'thingWithMergeProducer','thingWithMergeProducer',
                                        'tryNoPut','tryNoPut'
    ),
    expectedLabelsAfterGet = cms.vstring('A','A','B','B','C','C','D','D','E','E','F','F',
                                         'G', 'G', 'H', 'H', 'I', 'I', 'J', 'J', 'K', 'K', 'L', 'L',
                                         'dependsOnThingToBeDropped1', 'dependsOnThingToBeDropped1',
                                         'm1', 'm1', 'm2', 'm2', 'm3', 'm3',
                                         'makeThingToBeDropped', 'makeThingToBeDropped',
                                         'makeThingToBeDropped1', 'makeThingToBeDropped1',
                                         'thingWithMergeProducer', 'thingWithMergeProducer'
    ),
    expectedNumberOfThingsWithLabelA = cms.uint32(2)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMerge11.root')
)

process.p1 = cms.Path((process.m1 + process.m2 + process.m3) *
                     process.thingWithMergeProducer *
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
                      process.L *
                      process.testGetterOfProducts *
                      process.testGetterOfProductsA *
                      process.testGetterOfProductsARun *
                      process.testGetterOfProductsALumi 
)

process.e = cms.EndPath(process.out)
