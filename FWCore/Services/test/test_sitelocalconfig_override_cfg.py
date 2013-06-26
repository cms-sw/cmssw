import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.tester = cms.EDAnalyzer("SiteLocalConfigServiceTester",
                            sourceTempDir=cms.untracked.string("/x/y/z"),
                            sourceCacheHint=cms.untracked.string("storage-only"),
                            sourceReadHint=cms.untracked.string("direct-unbuffered"),
                            sourceTTreeCacheSize=cms.untracked.uint32(0),
                            sourceNativeProtocols=cms.untracked.vstring("rfio"),
                            sourceValuesSet=cms.untracked.bool(True)
)

process.o = cms.EndPath(process.tester)

process.add_(cms.Service("SiteLocalConfigService",
                         overrideSourceCacheTempDir=cms.untracked.string("/x/y/z"),
                         overrideSourceCacheHintDir=cms.untracked.string("storage-only"),
                         overrideSourceReadHint=cms.untracked.string("direct-unbuffered"),
                         overrideSourceNativeProtocols=cms.untracked.vstring("rfio"),
                         overrideSourceTTreeCacheSize=cms.untracked.uint32(0)))
