import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:refconsistency_1.root",
                                      "file:refconsistency_10.root")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("refconsistency_merge.root")
)

process.tester = cms.EDAnalyzer("OtherThingAnalyzer",
    other = cms.untracked.InputTag("otherThing","testUserTag")
)

process.o = cms.EndPath(process.out+process.tester)

