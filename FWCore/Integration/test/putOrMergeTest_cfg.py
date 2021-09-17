import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PutOrMergeTestSource")


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

    expectedBeginRunProd = cms.untracked.vint32(
    ),

    expectedEndRunProd = cms.untracked.vint32(
        100001,   200004,  100003,
    ),

    expectedEndRunProdImproperlyMerged = cms.untracked.vint32(
    )
)

#process.dump = cms.EDAnalyzer("EventContentAnalyzer")
process.end = cms.EndPath(process.test)

#process.add_(cms.Service("Tracer"))
