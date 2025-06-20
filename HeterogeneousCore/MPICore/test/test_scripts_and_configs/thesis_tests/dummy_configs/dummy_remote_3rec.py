import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("DUMMYREMOTE")

# Source: receive from MPI
process.source = cms.Source("MPISource",
    run_local=cms.untracked.bool(True),
    firstRun=cms.untracked.uint32(1)
)
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))

# Concurrency
process.options.numberOfThreads = int(os.environ.get("EXPERIMENT_THREADS", 2))
process.options.numberOfStreams = int(os.environ.get("EXPERIMENT_STREAMS", 2))
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.wantSummary = False


# Logging
process.MessageLogger = cms.Service("MessageLogger",
    cerr=cms.untracked.PSet(enableStatistics=cms.untracked.bool(False)),
    MPIReceiver=cms.untracked.PSet()
)

# FastTimer output
experiment_name = os.environ.get("EXPERIMENT_NAME", "unnamed")
output_dir = os.environ.get("EXPERIMENT_OUTPUT_DIR", "../../test_results/one_time_tests/")

process.FastTimerService = cms.Service( "FastTimerService",
    printEventSummary = cms.untracked.bool( False ),
    printRunSummary = cms.untracked.bool( True ),
    printJobSummary = cms.untracked.bool( True ),
    writeJSONSummary = cms.untracked.bool( False ),
    jsonFileName = cms.untracked.string( "resources.json" ),
    enableDQM = cms.untracked.bool( True ),
    enableDQMbyModule = cms.untracked.bool( False ),
    enableDQMbyPath = cms.untracked.bool( False ),
    enableDQMbyLumiSection = cms.untracked.bool( True ),
    enableDQMbyProcesses = cms.untracked.bool( True ),
    enableDQMTransitions = cms.untracked.bool( False ),
    dqmTimeRange = cms.untracked.double( 2000.0 ),
    dqmTimeResolution = cms.untracked.double( 5.0 ),
    dqmMemoryRange = cms.untracked.double( 1000000.0 ),
    dqmMemoryResolution = cms.untracked.double( 5000.0 ),
    dqmPathTimeRange = cms.untracked.double( 100.0 ),
    dqmPathTimeResolution = cms.untracked.double( 0.5 ),
    dqmPathMemoryRange = cms.untracked.double( 1000000.0 ),
    dqmPathMemoryResolution = cms.untracked.double( 5000.0 ),
    dqmModuleTimeRange = cms.untracked.double( 40.0 ),
    dqmModuleTimeResolution = cms.untracked.double( 0.2 ),
    dqmModuleMemoryRange = cms.untracked.double( 100000.0 ),
    dqmModuleMemoryResolution = cms.untracked.double( 500.0 ),
    dqmLumiSectionsRange = cms.untracked.uint32( 2500 ),
    dqmPath = cms.untracked.string( "HLT/TimerService" ),
)

process.ThroughputService = cms.Service( "ThroughputService",
    eventRange = cms.untracked.uint32( 10000 ),
    eventResolution = cms.untracked.uint32( 50 ),
    printEventSummary = cms.untracked.bool( False ),
    enableDQM = cms.untracked.bool( True ),
    dqmPathByProcesses = cms.untracked.bool( True ),
    dqmPath = cms.untracked.string( "HLT/Throughput" ),
    timeRange = cms.untracked.double( 60000.0 ),
    timeResolution = cms.untracked.double( 5.828 )
)

process.FastTimerService.writeJSONSummary = True
process.FastTimerService.jsonFileName=cms.untracked.string(f"{output_dir}/remote_{experiment_name}.json")

# MPI service
process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.MPIService.pmix_server_uri = "file:server.uri"

# Receiver
process.remoteDummyReceiver1 = cms.EDProducer("MPIReceiver",
    upstream=cms.InputTag("source"),
    instance=cms.int32(1),
    products=cms.VPSet(
        cms.PSet(
            type=cms.string("edm::FixedSizeDummy"),
            label=cms.string("")
        )
    )
)

process.remoteDummyReceiver2 = cms.EDProducer("MPIReceiver",
    upstream=cms.InputTag("source"),
    instance=cms.int32(2),
    products=cms.VPSet(
        cms.PSet(
            type=cms.string("edm::FixedSizeDummy"),
            label=cms.string("")
        )
    )
)

process.remoteDummyReceiver3 = cms.EDProducer("MPIReceiver",
    upstream=cms.InputTag("source"),
    instance=cms.int32(3),
    products=cms.VPSet(
        cms.PSet(
            type=cms.string("edm::FixedSizeDummy"),
            label=cms.string("")
        )
    )
)

# Path
process.remotePath = cms.Path(
    process.remoteDummyReceiver1 +
    process.remoteDummyReceiver2 +
    process.remoteDummyReceiver3)
process.schedule = cms.Schedule(process.remotePath)
