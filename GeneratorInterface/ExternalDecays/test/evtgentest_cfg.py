import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    )
)

#Before this test run for example Py6EvtGenFilter_cfg.py to produce the file
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/TestEvtGen.root')
)

process.Test = cms.EDAnalyzer("EvtGenTestAnalyzer",
    HistOutFile = cms.untracked.string('Test.root')
)

process.p1 = cms.Path(process.Test)
