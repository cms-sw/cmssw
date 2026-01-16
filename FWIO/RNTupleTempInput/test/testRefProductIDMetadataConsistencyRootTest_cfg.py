import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.source = cms.Source("RNTupleTempSource",
    fileNames = cms.untracked.vstring("file:refconsistency_merge.root")
)

process.tester = cms.EDAnalyzer("OtherThingAnalyzer",
    other = cms.untracked.InputTag("otherThing","testUserTag")
)

process.e = cms.EndPath(process.tester)
