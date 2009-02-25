import FWCore.ParameterSet.Config as cms

source = cms.Source("WatcherSource",
                    inputDir      = cms.string('in'),
                    filePatterns  = cms.vstring('.*\\.dat$'),
                    inprocessDir  = cms.string('work'),
                    processedDir  = cms.string('done'),
                    corruptedDir  = cms.string('corrupted'),
                    tokenFile     = cms.untracked.string('tok'),
                    verbosity     = cms.untracked.int32(1)
)
