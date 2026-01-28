import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIController")

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
# MPIController supports a single concurrent LuminosityBlock
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.numberOfConcurrentRuns = 1
process.options.wantSummary = False

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")

from eventlist_cff import eventlist
process.source = cms.Source("EmptySourceFromEventIDs",
    events = cms.untracked(eventlist)
)

process.maxEvents.input = 30

from HeterogeneousCore.MPICore.modules import MPIController, MPISender, MPIReceiver

process.mpiController = MPIController(
    mode = 'CommWorld'
)

# Produce EventID, that will be sent to the Follower
process.eventIDProducer = cms.EDProducer("edmtest::EventIDProducer")

# Send the EventID to the Follower. The Follower will run a ModuloEventIDFilter
# and send back the results.
process.sender = MPISender(
    upstream = "mpiController",
    instance = 42,
    products = [ "edmEventID_eventIDProducer__*" ]
)

# Receive a PathStateToken back from the Follower. This token will be present if
# the remore filter passed, and missing otherwise (see follower_filters_cfg.py for
# details)
process.receiver = MPIReceiver(
    upstream = "sender",
    instance = 99,
    products = [
        dict(
            type = "edm::PathStateToken",
            label = "remoteCapture"
        )
    ]
)

# The PathStateRelease module below is a filter that will pass if the
# edm::PathStateToken above is present, and will not pass otherwise.
from FWCore.Modules.modules import PathStateRelease
process.remoteRelease = PathStateRelease(
    state = cms.InputTag("receiver", "remoteCapture")
)

# The Follower runs "ModuloEventIDFilter" which accepts only events where (event
# number % `modulo` == 0). This is configured in follower_filters_cfg.py. In order
# to replicate the remote filtering locally, the same `modulo` value must be used
# here.
modulo = 3

# Controller runs the same filter locally
process.localFilter = cms.EDFilter("ModuloEventIDFilter",
    modulo = cms.uint32(modulo),
    offset = cms.uint32(0)
)

# Local filter path
process.localFilterPath = cms.Path(
    process.localFilter        # Apply the eventID filter locally
)

process.controllerPath = cms.Path(
    process.mpiController +    # Set up the MPI communication
    process.eventIDProducer +  # Produce EventID
    process.sender +           # Send EventID to the Follower
    process.receiver           # Receive PathStateToken back from the Follower
)

# Path that will pass (fail) if the remote filter path passed (failed)
process.remoteFilterPath = cms.Path(
    process.remoteRelease
)


# Compare local filter path and remote filter path. The outcome of these paths
# should be the same.
from FWCore.Modules.modules import PathStatusFilter
process.compare = PathStatusFilter(
    logicalExpression = '(localFilterPath and remoteFilterPath) or (not localFilterPath and not remoteFilterPath)'
)

process.validatePath = cms.Path(
    process.compare
)

# Verify that all events passed the comparison
from FWCore.Framework.modules import SewerModule
process.require = SewerModule(
    name = cms.string('require'),
    shouldPass = process.maxEvents.input.value(),
    SelectEvents = dict(
        SelectEvents = 'validatePath'
    )
)

process.endpath = cms.EndPath(
    process.require
)
