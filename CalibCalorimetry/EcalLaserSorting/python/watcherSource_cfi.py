import FWCore.ParameterSet.Config as cms

source = cms.Source("WatcherSource",
                    inputDir      = cms.string('in/'),
                    filePatterns  = cms.vstring('run[[:digit:]]+/run.*\\\\.dat$'),
#                    filePatterns  = cms.vstring('run.*\\\\.dat$'),
                    inprocessDir  = cms.string('work'),
                    processedDir  = cms.string('done'),
                    corruptedDir  = cms.string('corrupted'),
                    tokenFile     = cms.untracked.string('tok'),
                    verbosity     = cms.untracked.int32(1),
#Waiting time out. Stop CMSSW if no file is found in the inputDir after this delay.
#There is no time out when starting the process. If there is no input file, when
#starting the process, it will wait forever. The time out applies only  once some files
#were processed and input directory got exhausted.
                    timeOutInSec  = cms.int32(4*60)
)
