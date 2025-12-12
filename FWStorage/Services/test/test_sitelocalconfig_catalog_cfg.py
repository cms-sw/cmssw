import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.tester = cms.EDAnalyzer("SiteLocalConfigServiceCatalogTester",
    files = cms.untracked.VPSet(
        cms.untracked.PSet(
            file = cms.untracked.string("/store/a/b.root"),
            catalogIndex = cms.untracked.uint32(0),
            expectResult = cms.untracked.string("root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/store/a/b.root")
        ),
        cms.untracked.PSet(
            file = cms.untracked.string("/store/a/b.root"),
            catalogIndex = cms.untracked.uint32(1),
            expectResult = cms.untracked.string("root://xrootd-cms.infn.it//store/a/b.root")
        ),
    )
)

process.o = cms.EndPath(process.tester)
