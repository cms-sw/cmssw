import FWCore.ParameterSet.Config as cms

process = cms.Process("COPY")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:myout.root')
)

process.a1 = cms.EDAnalyzer("StreamThingAnalyzer",
    product_to_get = cms.string('m1')
)

process.out = cms.OutputModule("EventStreamFileWriter",
    fileName = cms.untracked.string('teststreamfile_copy.dat'),
    compression_level = cms.untracked.int32(1),
    use_compression = cms.untracked.bool(True),
    indexFileName = cms.untracked.string('testindexfile_copy.ind'),
    max_event_size = cms.untracked.int32(7000000)
)

process.outp = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myout2.root')
)

process.e = cms.EndPath(process.a1*process.out*process.outp)
