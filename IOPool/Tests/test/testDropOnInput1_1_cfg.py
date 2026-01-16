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

process.prodA = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag()
)

process.prodB = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("prodA"),
    onlyGetOnEvent = cms.untracked.uint32(1)
)

process.prodC = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("prodB"),
    onlyGetOnEvent = cms.untracked.uint32(2)
)

process.prodF = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag()
)

process.prodG = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("prodF"),
    onlyGetOnEvent = cms.untracked.uint32(2)
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

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testDropOnInput1_1.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_NK101_*_*',
        'drop *_A201_*_*',
        'drop *_prod2_*_*'
    )
)

process.path1 = cms.Path(process.prod1 + process.prod2 + process.prod3)
process.path2 = cms.Path(process.prodA + process.prodB + process.prodC)
process.path3 = cms.Path(process.prodF + process.prodG)
process.path4 = cms.Path(process.K100 + process.NK101 + process.K102 + process.K104)
process.path5 = cms.Path(process.K200 + process.K201 + process.K202 + process.K204)

process.endpath = cms.EndPath(process.out)
