import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

process = cms.Process("Test")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 1

process.options.numberOfThreads = 2
process.options.numberOfStreams = 0

process.dummySyncAna = cms.EDAnalyzer("SonicDummyOneAnalyzer",
    input = cms.int32(1),
    expected = cms.int32(-1),
    Client = cms.PSet(
        mode = cms.string("Sync"),
        factor = cms.int32(-1),
        wait = cms.int32(10),
        allowedTries = cms.untracked.uint32(0),
        fails = cms.uint32(0),
    ),
)

process.dummySyncAnaRetry = process.dummySyncAna.clone(
    Client = dict(
        wait = 2,
        allowedTries = 2,
        fails = 1,
    )
)

process.p1 = cms.Path(process.dummySyncAna)
process.p2 = cms.Path(process.dummySyncAnaRetry)
