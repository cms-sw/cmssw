import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

#process.source = cms.Source("EmptySource")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      '/store/data/Run2016C/DoubleMuon/MINIAOD/02Feb2017-v2/70000/7003FC2E-C6E9-E611-B741-0CC47A7C3610.root',
#      '/store/data/Run2016C/DoubleMuon/MINIAOD/02Feb2017-v2/70000/2E2F8BF3-C4E9-E611-B125-0025905B8576.root'
      )
    )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.tester = cms.EDAnalyzer("SiteLocalConfigServiceTester",
                            sourceTempDir=cms.untracked.string("/a/b/c"),
                            sourceCacheHint=cms.untracked.string("application-only"),
                            sourceReadHint=cms.untracked.string("read-ahead-buffered"),
                            sourceTTreeCacheSize=cms.untracked.uint32(10000),
                            sourceNativeProtocols=cms.untracked.vstring("dcache","file"),
                            sourceValuesSet=cms.untracked.bool(True)
)

process.o = cms.EndPath(process.tester)
