# /online/collisions/2024/2e34/v1.4/HLT/V2 (CMSSW_14_0_11)

import FWCore.ParameterSet.Config as cms

# load the "frozen" 2024 HLT menu
from hlt_cff import process

# run over HLTPhysics data from run 383363
process.load('run383631_cff')

# override the GlobalTag
from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
process.GlobalTag = customiseGlobalTag(process.GlobalTag, globaltag = '141X_dataRun3_HLT_v1')

# update the HLT menu for re-running offline using a recent release
from HLTrigger.Configuration.customizeHLTforCMSSW import customizeHLTforCMSSW
process = customizeHLTforCMSSW(process)

# create the DAQ working directory for DQMFileSaverPB
import os
os.makedirs('%s/run%d' % (process.EvFDaqDirector.baseDir.value(), process.EvFDaqDirector.runNumber.value()), exist_ok=True)

# run with 32 threads, 24 concurrent events, 2 concurrent lumisections, over 10k events
process.options.numberOfThreads = 32
process.options.numberOfStreams = 24
process.options.numberOfConcurrentLuminosityBlocks = 2
process.maxEvents.input = 10300

# force the '2e34' prescale column
process.PrescaleService.lvl1DefaultLabel = '2p0E34'
process.PrescaleService.forceDefault = True

# do not print a final summary
process.options.wantSummary = False
process.MessageLogger.cerr.enableStatistics = cms.untracked.bool(False)

# write a JSON file with the timing information
process.FastTimerService.writeJSONSummary = True

process.ThroughputService = cms.Service('ThroughputService',
    enableDQM = cms.untracked.bool(False),
    printEventSummary = cms.untracked.bool(True),
    eventResolution = cms.untracked.uint32(100),
    eventRange = cms.untracked.uint32(10300),
)

process.MessageLogger.cerr.ThroughputService = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000),
    reportEvery = cms.untracked.int32(1)
)
