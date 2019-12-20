import FWCore.ParameterSet.Config as cms

process = cms.Process("TRANSFER")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("WatcherSource",
                            inputDir = cms.string("."),
                            filePatterns = cms.vstring("inDir/.*\.dat"),
                            inprocessDir = cms.string("process"),
                            processedDir = cms.string("processed"),
                            corruptedDir = cms.string("corrupt"),
                            tokenFile = cms.untracked.string("watcherSourceToken"),
                            timeOutInSec = cms.int32(10),
                            verbosity = cms.untracked.int32(1)
)

#process.finishProcessFile = cms.EDAnalyzer("ecallasersortingtest::CreateFileAfterStartAnalyzer",
#                              fileName = cms.untracked.string("watcherSourceToken")
#)

process.a1 = cms.EDAnalyzer("StreamThingAnalyzer",
    product_to_get = cms.string('m1')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myout.root')
)

process.end = cms.EndPath(process.a1*process.out)
