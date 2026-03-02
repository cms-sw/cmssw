import FWCore.ParameterSet.Config as cms

process = cms.Process("COPYDROP")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:goodDataFormatsFWLite.root')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('refTestCopyDropDataFormatsFWLite.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)

process.outp = cms.EndPath(process.out)
