import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST2" )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testSubProcessUnscheduled.root'
    )
)

process.test = cms.EDAnalyzer("TestParentage",
                                 inputTag = cms.InputTag("final"),
                                 expectedAncestors = cms.vstring("two", "ten", "adder")
)
process.o = cms.EndPath(process.test)

# foo bar baz
# sO57hFqU5m4YH
# u0Xpq7cqoQy2v
