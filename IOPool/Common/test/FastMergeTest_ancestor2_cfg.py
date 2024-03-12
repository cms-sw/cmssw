# The following comments couldn't be translated into the new config version:

# Configuration file for FastMergeTest

import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTMERGE")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.AdaptorConfig = cms.Service("AdaptorConfig",
    stats = cms.untracked.bool(False)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:FastMerge_ancestor_out.root')
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:FastMergeTest_ancestor2.root', 
        'file:FastMergeTest_ancestor1.root')
)

process.ep = cms.EndPath(process.output)


# foo bar baz
# 0pP74y5fv81ot
# fNWms9QgSJvjL
