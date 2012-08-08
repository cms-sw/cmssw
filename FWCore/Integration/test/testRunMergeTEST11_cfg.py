import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST11")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold = 'ERROR'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        # CAUTION if you recreate the PROD files then you must recreate BOTH
        # of these files otherwise you will get exceptions because the GUIDs
        # used to check the match of the event in the secondary files will
        # not be the same.
        'file:testRunMerge.root',
        'file:testRunMergeMERGE2.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
        'file:testRunMerge0.root', 
        'file:testRunMerge1.root', 
        'file:testRunMerge2.root', 
        'file:testRunMerge3.root',
        'file:testRunMerge4.root',
        'file:testRunMerge5.root'
    )
    , duplicateCheckMode = cms.untracked.string('checkEachRealDataFile')
    , noEventSort = cms.untracked.bool(False)
)

process.testGetterOfProducts = cms.EDFilter("TestGetterOfProducts",
    processName = cms.string('PROD'),
    expectedInputTagLabels = cms.vstring('A','B','C','D','E','F', 'G', 'H', 'I', 'J', 'K', 'L',
                                        'dependsOnThingToBeDropped1',
                                        'm1', 'm2', 'm3',
                                        'makeThingToBeDropped',
                                        'makeThingToBeDropped1',
                                        'thingWithMergeProducer',
                                        'tryNoPut'
                                        ),
    expectedLabelsAfterGet = cms.vstring('A','B','C','D','E','F', 'G', 'H', 'I', 'J', 'K', 'L',
                                         'dependsOnThingToBeDropped1',
                                         'm1', 'm2', 'm3',
                                         'makeThingToBeDropped',
                                         'thingWithMergeProducer'
    )
)

process.testGetterOfProductsA = cms.EDAnalyzer("TestGetterOfProductsA",
    processName = cms.string('PROD'),
    branchType = cms.int32(0),
    expectedInputTagLabels = cms.vstring('A','B','C','D','E','F', 'G', 'H', 'I', 'J', 'K', 'L',
                                        'dependsOnThingToBeDropped1',
                                        'm1', 'm2', 'm3',
                                        'makeThingToBeDropped',
                                        'makeThingToBeDropped1',
                                        'thingWithMergeProducer',
                                        'tryNoPut'
                                        ),
    expectedLabelsAfterGet = cms.vstring('A','B','C','D','E','F', 'G', 'H', 'I', 'J', 'K', 'L',
                                         'dependsOnThingToBeDropped1',
                                         'm1', 'm2', 'm3',
                                         'makeThingToBeDropped',
                                         'thingWithMergeProducer'
    ),
    expectedNumberOfThingsWithLabelA = cms.uint32(1)
)

process.testGetterOfProductsALumi = cms.EDAnalyzer("TestGetterOfProductsA",
    processName = cms.string('PROD'),
    branchType = cms.int32(1),
    expectedInputTagLabels = cms.vstring('A','A','B','B','C','C','D','D','E','E','F','F', 'G','G', 'H','H', 'I','I', 'J','J', 'K','K', 'L','L',
                                        'dependsOnThingToBeDropped1','dependsOnThingToBeDropped1',
                                        'm1','m1', 'm2','m2', 'm3','m3',
                                        'makeThingToBeDropped','makeThingToBeDropped',
                                        'makeThingToBeDropped1','makeThingToBeDropped1',
                                        'thingWithMergeProducer','thingWithMergeProducer',
                                        'tryNoPut','tryNoPut'
                                        ),
    expectedLabelsAfterGet = cms.vstring('A','A','B','B','C','C','D','D','E','E','F','F', 'G','G', 'H','H', 'I','I', 'J','J', 'K','K', 'L','L',
                                         'dependsOnThingToBeDropped1','dependsOnThingToBeDropped1',
                                         'm1','m1', 'm2','m2', 'm3','m3',
                                         'makeThingToBeDropped','makeThingToBeDropped',
                                         'makeThingToBeDropped1','makeThingToBeDropped1',
                                         'thingWithMergeProducer','thingWithMergeProducer'
    ),
    expectedNumberOfThingsWithLabelA = cms.uint32(2)
)

process.testGetterOfProductsARun = cms.EDAnalyzer("TestGetterOfProductsA",
    processName = cms.string('PROD'),
    branchType = cms.int32(2),
    expectedInputTagLabels = cms.vstring('A','A','B','B','C','C','D','D','E','E','F','F', 'G','G', 'H','H', 'I','I', 'J','J', 'K','K', 'L','L',
                                        'dependsOnThingToBeDropped1','dependsOnThingToBeDropped1',
                                        'm1','m1', 'm2','m2', 'm3','m3',
                                        'makeThingToBeDropped','makeThingToBeDropped',
                                        'makeThingToBeDropped1','makeThingToBeDropped1',
                                        'thingWithMergeProducer','thingWithMergeProducer',
                                        'tryNoPut','tryNoPut'
                                        ),
    expectedLabelsAfterGet = cms.vstring('A','A','B','B','C','C','D','D','E','E','F','F', 'G','G', 'H','H', 'I','I', 'J','J', 'K','K', 'L','L',
                                         'dependsOnThingToBeDropped1','dependsOnThingToBeDropped1',
                                         'm1','m1', 'm2','m2', 'm3','m3',
                                         'makeThingToBeDropped','makeThingToBeDropped',
                                         'makeThingToBeDropped1','makeThingToBeDropped1',
                                         'thingWithMergeProducer','thingWithMergeProducer'
    ),
    expectedNumberOfThingsWithLabelA = cms.uint32(2)
)

process.path1 = cms.Path(process.testGetterOfProducts*process.testGetterOfProductsA*process.testGetterOfProductsALumi*process.testGetterOfProductsARun)
