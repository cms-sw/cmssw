# Configuration file for testing files with differing product ids   

import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.source = cms.Source("PoolSource",
    # hack until metadata pruning is implemented
    dropMetaData = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring('file:prod1.root', 
        'file:prod2.root')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('prodmerge.root')
)

process.outp = cms.EndPath(process.out)
