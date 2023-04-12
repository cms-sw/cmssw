import FWCore.ParameterSet.Config as cms

process = cms.Process( "DUMP" )

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:PoolOutputTest.root")
)

process.dump = cms.EDAnalyzer("DumpMuonScouting",
    muInputTag = cms.InputTag("rawDataCollector"),
    minBx = cms.int32(0),
    maxBx = cms.int32(4000)
)


process.p = cms.Path(
  process.dump
)

