import FWCore.ParameterSet.Config as cms
import os

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

# environment-configurable parameters
threads = int(os.environ.get("EXPERIMENT_THREADS", 32))
streams = int(os.environ.get("EXPERIMENT_STREAMS", 24))
experiment_name = os.environ.get("EXPERIMENT_NAME", "local_test")
output_dir = os.environ.get("EXPERIMENT_OUTPUT_DIR", "./test_results/local_pipeline/")

# threading configuration
process.options.numberOfThreads = threads
process.options.numberOfStreams = streams
process.options.numberOfConcurrentLuminosityBlocks = 2
process.maxEvents.input = 10000 #1k

# create the DAQ working directory for DQMFileSaverPB
os.makedirs('%s/run%d' % (process.EvFDaqDirector.baseDir.value(), process.EvFDaqDirector.runNumber.value()), exist_ok=True)

# force the '2e34' prescale column
process.PrescaleService.lvl1DefaultLabel = '2p0E34'
process.PrescaleService.forceDefault = True

# logging
process.options.wantSummary = False

# timing output
process.FastTimerService.writeJSONSummary = True
process.FastTimerService.jsonFileName = cms.untracked.string(f"{output_dir}/local_{experiment_name}.json")
