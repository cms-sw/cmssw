import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST1")

process.source = cms.Source("EmptySource",
    firstEvent = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.prod1 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag()
)

process.prod2 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("prod1")
)

process.prod3 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("prod2")
)

process.K100 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag()
)

process.NK101 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("K100")
)

process.A101 = cms.EDAlias(NK101 = cms.VPSet( cms.PSet(type=cms.string('edmtestIntProduct') ) ) )

process.K102 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("NK101")
)

process.K104 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("A101")
)

process.K200 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag()
)

process.K201 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("K200")
)

process.A201 = cms.EDAlias(K201 = cms.VPSet( cms.PSet(type=cms.string('edmtestIntProduct') ) ) )

process.K202 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("K201")
)

process.K204 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("A201")
)

process.path1 = cms.Path(process.prod1 + process.prod2 + process.prod3)
process.path4 = cms.Path(process.K100 + process.NK101 + process.K102 + process.K104)
process.path5 = cms.Path(process.K200 + process.K201 + process.K202 + process.K204)

# ---------------------------------------------------------------

subProcess1 = cms.Process("TESTSUB1")
process.addSubProcess(cms.SubProcess(subProcess1,
            outputCommands = cms.untracked.vstring(
                'keep *',
                'drop *_NK101_*_*',
                'drop *_A201_*_*',
                'drop *_prod2_*_*'
        )
    )
)

subProcess1.K103 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("A101")
)

subProcess1.K203 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("K201")
)

subProcess1.test1 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("A101"),
                               expectedAncestors = cms.vstring("K100"),
                               callGetProvenance = cms.untracked.bool(False)
)

subProcess1.test2 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("K100"),
                               expectedAncestors = cms.vstring()
)

subProcess1.test3 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("K102"),
                               expectedAncestors = cms.vstring("K100", "NK101"),
                               callGetProvenance = cms.untracked.bool(False)
)

subProcess1.test4 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("K103"),
                               expectedAncestors = cms.vstring("K100", "NK101"),
                               callGetProvenance = cms.untracked.bool(False)
)

subProcess1.test5 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("K104"),
                               expectedAncestors = cms.vstring("K100", "NK101"),
                               callGetProvenance = cms.untracked.bool(False)
)

subProcess1.path1 = cms.Path(subProcess1.K103 * subProcess1.K203)
subProcess1.endpath1 = cms.EndPath(subProcess1.test1 *
                                   subProcess1.test2 *
                                   subProcess1.test3 *
                                   subProcess1.test4 *
                                   subProcess1.test5
)

subProcess2 = cms.Process("TESTSUB2")
subProcess1.addSubProcess(cms.SubProcess(subProcess2,
            outputCommands = cms.untracked.vstring(
                'keep *'
        )
    )
)

subProcess2.K105 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("K103")
)

subProcess2.K205 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("K203")
)



subProcess2.test1 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("A101"),
                               expectedAncestors = cms.vstring("K100"),
                               callGetProvenance = cms.untracked.bool(False)
)

subProcess2.test2 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("K100"),
                               expectedAncestors = cms.vstring()
)

subProcess2.test3 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("K102"),
                               expectedAncestors = cms.vstring("K100", "NK101"),
                               callGetProvenance = cms.untracked.bool(False)
)

subProcess2.test4 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("K103"),
                               expectedAncestors = cms.vstring("K100", "NK101"),
                               callGetProvenance = cms.untracked.bool(False)
)

subProcess2.test5 = cms.EDAnalyzer("TestParentage",
                               inputTag = cms.InputTag("K104"),
                               expectedAncestors = cms.vstring("K100", "NK101"),
                               callGetProvenance = cms.untracked.bool(False)
)

subProcess2.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testDropOnInputSubProcess.root'),
    outputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_NK*_*_*'
    )
)

subProcess2.path1 = cms.Path(subProcess2.K105 * subProcess2.K205)

subProcess2.endpath1 = cms.EndPath(subProcess2.out *
                                   subProcess2.test1 *
                                   subProcess2.test2 *
                                   subProcess2.test3 *
                                   subProcess2.test4 *
                                   subProcess2.test5
)
