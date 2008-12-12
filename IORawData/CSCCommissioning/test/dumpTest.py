import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/FEBAD3FE-F69D-DD11-A12B-001D09F2906A.root')
)

process.cscdumper = cms.EDAnalyzer("CSCFileDumper",
    output = cms.untracked.string("/tmp/kkotov/rawdata.bin")
)

process.p = cms.Path(process.cscdumper)

