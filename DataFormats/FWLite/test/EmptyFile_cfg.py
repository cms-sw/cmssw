import FWCore.ParameterSet.Config as cms

process = cms.Process("NOCOPY")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:good.root'),
    firstRun = cms.untracked.uint32(100)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('empty.root')
)

process.outp = cms.EndPath(process.out)
