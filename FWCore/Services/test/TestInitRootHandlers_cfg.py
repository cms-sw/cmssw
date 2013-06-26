# Configuration file for TestInitRootHandlers

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.MessageLogger = cms.Service("MessageLogger",
    TestInitRootHandlers = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noTimeStamps = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('TestInitRootHandlers')
)

process.InitRootHandlers = cms.Service("InitRootHandlers")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("EmptySource")

process.Thing = cms.EDProducer("ThingProducer")

process.OtherThing = cms.EDProducer("OtherThingProducer")

process.Analysis = cms.EDAnalyzer("TestInitRootHandlers",
    debugLevel = cms.untracked.int32(1)
)

process.p = cms.Path(process.Thing*process.OtherThing*process.Analysis)
