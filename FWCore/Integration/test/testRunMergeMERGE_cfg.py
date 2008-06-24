import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

import FWCore.Framework.cmsExceptionsFatal_cff
process.options = FWCore.Framework.cmsExceptionsFatal_cff.options

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge1.root', 
        'file:testRunMerge2.root', 
        'file:testRunMerge3.root'
    )
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")

process.test = cms.EDFilter("TestMergeResults",
                            
    #   These values below are just arbitrary and meaningless
    #   We are checking to see that the value we get out matches what
    #   was put in.

    #   expected values listed below come in sets of three
    #      value expected in Thing
    #      value expected in ThingWithMerge
    #      value expected in ThingWithIsEqual
    #   This set of 3 is repeated below at each point it might change
    #   The Prod suffix refers to objects from the process named PROD
    #   The New suffix refers to objects created in the most recent process
    #   When the sequence of parameter values is exhausted it stops checking
    #   0's are just placeholders, if the value is a "0" the check is not made
    #   and it indicates the product does not exist at that point.
    #   *'s indicate lines where the checks are actually run by the test module.

    expectedBeginRunProd = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        10001,   10002,  10003,   # * begin run 1
        10001,   20004,  10003,   # * begin file 2
        10001,   20004,  10003,   # end run 1
        10001,   10002,  10003,   # * begin run 2
        10001,   10002,  10003,   # begin file 3
        10001,   10002,  10003,   # end run 2
        10001,   30006,  10003,   # * begin run 1
        10001,   30006,  10003    # end run 1
    ),
                            
    expectedEndRunProd = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        100001, 100002, 100003,   # * begin run 1
        100001, 200004, 100003,   # * begin file 2
        100001, 200004, 100003,   # * end run 1
        100001, 100002, 100003,   # * begin run 2
        100001, 100002, 100003,   # begin file 3
        100001, 100002, 100003,   # * end run 2
        100001, 300006, 100003,   # * begin run 1
        100001, 300006, 100003    # * end run 1
    ),

    expectedBeginLumiProd = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        101,       102,    103,   # * begin run 1 lumi 1
        101,       204,    103,   # * begin file 2
        101,       204,    103,   # end run 1 lumi 1
        101,       102,    103,   # * begin run 2 lumi 1
        101,       102,    103,   # begin file 3
        101,       102,    103,   # end run 2 lumi 1
        101,       306,    103,   # * begin run 1 lumi 1
        101,       306,    103    # end run 1 lumi 1
    ),

    expectedEndLumiProd = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        1001,     1002,   1003,   # * begin run 1 lumi 1
        1001,     2004,   1003,   # * begin file 2
        1001,     2004,   1003,   # * end run 1 lumi 1
        1001,     1002,   1003,   # * begin run 2 lumi 1
        1001,     1002,   1003,   # begin file 3
        1001,     1002,   1003,   # * end run 2 lumi 1
        1001,     3006,   1003,   # * begin run 1 lumi 1
        1001,     3006,   1003    # * end run 1 lumi 1
    ),

    expectedBeginRunNew = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        10001,   10002,  10003,   # * begin run 1
        10001,   10002,  10003,   # * begin file 2
        10001,   10002,  10003,   # end run 1
        10001,   10002,  10003,   # * begin run 2
        10001,   10002,  10003,   # begin file 3
        10001,   10002,  10003,   # end run 2
        10001,   20004,  10003,   # * begin run 1
        10001,   20004,  10003    # end run 1
    ),

    expectedEndRunNew = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        0,           0,      0,   # * begin run 1
        0,           0,      0,   # * begin file 2
        100001, 100002, 100003,   # * end run 1
        0,           0,      0,   # * begin run 2
        0,           0,      0,   # begin file 3
        100001, 100002, 100003,   # * end run 2
        100001, 100002, 100003,   # * begin run 1
        100001, 200004, 100003    # * end run 1
    ),

    expectedBeginLumiNew = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        101,       102,    103,   # * begin run 1 lumi 1
        101,       102,    103,   # * begin file 2
        101,       102,    103,   # end run 1 lumi 1
        101,       102,    103,   # * begin run 2 lumi 1
        101,       102,    103,   # begin file 3
        101,       102,    103,   # end run 2 lumi 1
        101,       204,    103,   # * begin run 1 lumi 1
        101,       204,    103    # end run 1 lumi 1
    ),

    expectedEndLumiNew = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        0,           0,      0,   # * begin run 1 lumi 1
        0,           0,      0,   # * begin file 2
        1001,     1002,   1003,   # * end run 1 lumi 1
        0,           0,      0,   # * begin run 2 lumi 1
        0,           0,      0,   # begin file 3
        1001,     1002,   1003,   # * end run 2 lumi 1
        1001,     1002,   1003,   # * begin run 1 lumi 1
        1001,     2004,   1003    # * end run 1 lumi 1
    ),

    expectedRespondToOpenInputFile = cms.untracked.int32(3),
    expectedRespondToCloseInputFile = cms.untracked.int32(3),
    expectedRespondToOpenOutputFiles = cms.untracked.int32(1),
    expectedRespondToCloseOutputFiles = cms.untracked.int32(1),

    expectedInputFileNames = cms.untracked.vstring(
        'file:testRunMerge1.root', 
        'file:testRunMerge2.root', 
        'file:testRunMerge3.root'
    ),

    verbose = cms.untracked.bool(False)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMerge.root'),
    outputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_makeThingToBeDropped_*_*'
    )
)

process.path1 = cms.Path(process.thingWithMergeProducer + process.test)
process.e = cms.EndPath(process.out)
