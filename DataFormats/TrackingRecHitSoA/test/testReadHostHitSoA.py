import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+sys.argv[1]))

process.testReadHostHitSoA = cms.EDAnalyzer("TestReadHostHitSoA",
    input = cms.InputTag("hitSoA", "", "WRITE"),
    hitSize = cms.uint32(2708)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testTrackSoAReader.root'),
    fastCloning = cms.untracked.bool(False)
)

process.path = cms.Path(process.testReadHostHitSoA)

process.endPath = cms.EndPath(process.out)

