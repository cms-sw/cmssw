import FWCore.ParameterSet.Config as cms

process = cms.Process("CluFP420Test")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Geometry.FP420CommonData.FP420GeometryXML_cfi")

process.load("RecoRomanPot.RecoFP420.FP420Cluster_cfi")

process.load("RecoRomanPot.RecoFP420.FP420Track_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(11),
        mix = cms.untracked.uint32(12345),
        VtxSmeared = cms.untracked.uint32(98765432)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(100),
    fileNames = cms.untracked.vstring('file:testDIevent.root')
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testCLTRevent.root')
)

process.Timing = cms.Service("Timing")

process.Tracer = cms.Service("Tracer")

process.p1 = cms.Path(process.FP420Cluster*process.FP420Track)
process.outpath = cms.EndPath(process.o1)
process.schedule = cms.Schedule(process.p1,process.outpath)

process.MessageLogger.cerr.default.limit = 10


