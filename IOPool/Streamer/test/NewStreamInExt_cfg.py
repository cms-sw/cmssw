import FWCore.ParameterSet.Config as cms

process = cms.Process("TRANSFER")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring('file:teststreamfile.dat', 'file:teststreamfile_ext.dat')
    #firstEvent = cms.untracked.uint64(10123456835)
)

process.a1 = cms.EDAnalyzer("StreamThingAnalyzer",
    product_to_get = cms.string('m1')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myout.root')
)

process.end = cms.EndPath(process.a1*process.out)
