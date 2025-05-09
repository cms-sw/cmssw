#! /bin/bash

CONFIG=/dev/CMSSW_14_0_0/GRun/V124

# extract the configuration from the database
hltConfigFromDB --configName $CONFIG > Timing_HLT_GRun.py

# customise the configuration for the timing measurements
cat >> Timing_HLT_GRun.py << @EOF
# run over recent data
process.load('source_cff')

# use the date global tag
process.GlobalTag.globaltag = '140X_dataRun3_HLT_v3'

# update the HLT menu for re-running offline using a recent release
from HLTrigger.Configuration.customizeHLTforCMSSW import customizeHLTforCMSSW
process = customizeHLTforCMSSW(process)

# run with 32 threads, 24 concurrent events, 2 concurrent lumisections, over 20k events
process.options.numberOfThreads = 32
process.options.numberOfStreams = 24
process.options.numberOfConcurrentLuminosityBlocks = 2
process.maxEvents.input = 20300

# force the '2e34' prescale column
process.PrescaleService.lvl1DefaultLabel = '2p0E34'
process.PrescaleService.forceDefault = True

# do not run the HLTAnalyzerEndpath
if 'HLTAnalyzerEndpath' in process.endpaths:
  del process.HLTAnalyzerEndpath

# do not print a final summary
process.options.wantSummary = False
process.MessageLogger.cerr.enableStatistics = cms.untracked.bool(False)

# write a JSON file with the timing information
process.FastTimerService.writeJSONSummary = True
@EOF
