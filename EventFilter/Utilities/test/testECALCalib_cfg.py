import FWCore.ParameterSet.Config as cms

process = cms.Process("TRANSFER")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("WatcherSource",
                            inputDir = cms.string("."),
                            filePatterns = cms.vstring("ecalInDir/.*\.dat"),
                            inprocessDir = cms.string("process"),
                            processedDir = cms.string("processed"),
                            corruptedDir = cms.string("corrupt"),
                            tokenFile = cms.untracked.string("watcherSourceToken"),
                            timeOutInSec = cms.int32(10),
                            verbosity = cms.untracked.int32(1)
)

