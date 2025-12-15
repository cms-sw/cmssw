import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.tester = cms.EDAnalyzer("SiteLocalConfigServiceTester",
                            sourceTempDir=cms.untracked.string(""),
                            sourceCacheHint=cms.untracked.string(""),
                            sourceReadHint=cms.untracked.string(""),
                            sourceTTreeCacheSize=cms.untracked.uint32(0),
                            sourceNativeProtocols=cms.untracked.vstring(),
                            sourceValuesSet=cms.untracked.bool(False),
                            expectedUseLocalConnectString = cms.untracked.bool(False),
                            expectedLocalConnectPrefix = cms.untracked.string(""),
                            expectedLocalConnectSuffix = cms.untracked.string("")
)

process.o = cms.EndPath(process.tester)
