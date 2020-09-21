import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.tester = cms.EDAnalyzer("SiteLocalConfigServiceTester",
                            sourceTempDir=cms.untracked.string("/a/b/c"),
                            sourceCacheHint=cms.untracked.string("application-only"),
                            sourceReadHint=cms.untracked.string("read-ahead-buffered"),
                            sourceTTreeCacheSize=cms.untracked.uint32(10000),
                            sourceNativeProtocols=cms.untracked.vstring("dcache","file"),
                            sourceValuesSet=cms.untracked.bool(True),
                            expectedUseLocalConnectString = cms.untracked.bool(True),
                            expectedLocalConnectPrefix = cms.untracked.string("Test:Prefix"),
                            expectedLocalConnectSuffix = cms.untracked.string("Test.Suffix")
)

process.o = cms.EndPath(process.tester)
