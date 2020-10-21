import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['cerr']
process.MessageLogger.statistics = []


process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:AssociationMapTest.root'
    )
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

process.p = cms.Path(process.test)
