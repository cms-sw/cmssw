import FWCore.ParameterSet.Config as cms
process = cms.Process("MERGE")

process.source = cms.Source("PoolSource",
                            fileNames =
cms.untracked.vstring("ref_merge_prod1.root",
                      "ref_merge_prod2.root")
                            )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName =
cms.untracked.string("ref_merge.root")
                               )

process.o = cms.EndPath(process.out)

# foo bar baz
# o0LYc7AB8hHLY
# qo4CX3pO62ycf
