import FWCore.ParameterSet.Config as cms

process = cms.Process("NOCOPY")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:goodDataFormatsFWLite.root'),
    # firstRun gives a file with no runs, lumis, or events
    # firstEvent gives a file with no events, but with runs and lumis
    # firstRun = cms.untracked.uint32(100)
    firstEvent = cms.untracked.uint32(100)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('emptyDataFormatsFWLite.root')
)

process.outp = cms.EndPath(process.out)
