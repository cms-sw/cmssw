import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['cerr']
process.MessageLogger.statistics = []
process.MessageLogger.fwkJobReports = []

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("EmptySource")

process.intvec1 = cms.EDProducer("IntVectorProducer",
    count = cms.int32(9),
    ivalue = cms.int32(11),
    delta = cms.int32(1)
)

process.intvec2 = cms.EDProducer("IntVectorProducer",
    count = cms.int32(9),
    ivalue = cms.int32(21),
    delta = cms.int32(1)
)

process.associationMapProducer = cms.EDProducer("AssociationMapProducer",
    inputTag1 = cms.InputTag("intvec1"),
    inputTag2 = cms.InputTag("intvec2")
)

process.test = cms.EDAnalyzer("AssociationMapAnalyzer",
    inputTag1 = cms.InputTag("intvec1"),
    inputTag2 = cms.InputTag("intvec2"),
    associationMapTag1 = cms.InputTag("associationMapProducer"),
    associationMapTag2 = cms.InputTag("associationMapProducer", "twoArg"),
    associationMapTag3 = cms.InputTag("associationMapProducer"),
    associationMapTag4 = cms.InputTag("associationMapProducer", "handleArg"),
    associationMapTag5 = cms.InputTag("associationMapProducer"),
    associationMapTag6 = cms.InputTag("associationMapProducer"),
    associationMapTag7 = cms.InputTag("associationMapProducer"),
    associationMapTag8 = cms.InputTag("associationMapProducer", "twoArg")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('AssociationMapTest.root')
)

process.p = cms.Path(process.intvec1 * process.intvec2 * process.associationMapProducer * process.test)

process.e = cms.EndPath(process.out)
