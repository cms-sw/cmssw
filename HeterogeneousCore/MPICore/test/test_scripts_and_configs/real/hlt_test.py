import FWCore.ParameterSet.Config as cms

# run over HLTPhysics data from run 383363
from hlt import process

threads = int(os.environ.get("EXPERIMENT_THREADS", 32))
streams = int(os.environ.get("EXPERIMENT_STREAMS", 24))
experiment_name = os.environ.get("EXPERIMENT_NAME", "local_test")
output_dir = os.environ.get("EXPERIMENT_OUTPUT_DIR", "./test_results/local_pipeline/")

# threading configuration
process.options.numberOfThreads = threads
process.options.numberOfStreams = streams
process.options.numberOfConcurrentLuminosityBlocks = 2
process.maxEvents.input = 1000 #1k

# do not print a final summary
process.options.wantSummary = False
process.MessageLogger.cerr.enableStatistics = cms.untracked.bool(False)

# run the HBHE local reconstruction
process.HLTLocalHBHE = cms.Path(
    process.hltGetRaw +
    process.hltHcalDigis +
    process.hltHcalDigisSoA +
    process.hltHbheRecoSoA +
    process.hltParticleFlowRecHitHBHESoA +
    process.hltParticleFlowClusterHBHESoA
)

# run the ECAL local reconstruction
process.HLTLocalECAL = cms.Path(
    process.hltGetRaw +
    process.hltEcalDigisSoA +
    process.hltEcalUncalibRecHitSoA
)

# schedule the reconstruction on every event
process.schedule.extend([
    process.HLTLocalHBHE,
    process.HLTLocalECAL
])

process.FastTimerService.writeJSONSummary = True
process.FastTimerService.jsonFileName = cms.untracked.string(f"{output_dir}/whole_{experiment_name}.json")

#process.Tracer = cms.Service("Tracer")
