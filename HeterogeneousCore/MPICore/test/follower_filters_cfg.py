import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIFollower")

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
process.options.numberOfConcurrentLuminosityBlocks = 2
process.options.numberOfConcurrentRuns = 2
process.options.wantSummary = False

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")

from HeterogeneousCore.MPICore.modules import MPISource, MPIReceiver, MPISender

process.source = MPISource()

process.maxEvents.input = -1

# Receive EventIDs from the Controller
process.receiver = MPIReceiver(
    upstream = "source",
    instance = 42,
    products = [ dict(
        type = "edm::EventID",
        label = ""
    )]
)

# Filter that accepts events whose eventID % `modulo` == 0
modulo = 3
process.remoteFilter = cms.EDFilter("ModuloEventIDFilter",
    modulo = cms.uint32(modulo),
    offset = cms.uint32(0)
)

# This producer will be scheduled after the filter above, and will produce a
# `edm::PathStateToken` if the filter passes. If filter does not pass, this module
# will not run, and MPISender will detect the missing token.
from FWCore.Modules.modules import PathStateCapture
process.remoteCapture = PathStateCapture()

# Send back the PathStateToken to the Controller. Filtered-out events will be
# detected by MPISender because for them, the PathStateToken will be missing. When
# sending the metadata of such events, the MPISender will set `productCount = -1`
# to indicate that the event was filtered out.
process.sender = MPISender(
    upstream = "receiver",
    instance = 99,
    products = [
        "edmPathStateToken_remoteCapture__*"
    ]
)

# Path for the filter (+ the PathStateCapture)
process.filterPath = cms.Path(
    process.remoteFilter +     # Apply the filter
    process.remoteCapture      # Capture path state (if filter passes)
)

# The MPI modules are put in a different path so that they are always scheduled.
process.mpiPath = cms.Path(
    process.receiver +         # Receive original EventIDs from controller
    process.sender             # Send back PathStateToken
)
