import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIController")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 10000000

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
# MPIController supports a single concurrent LuminosityBlock
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.numberOfConcurrentRuns = 1
process.options.wantSummary = False

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")

process.load("Configuration.StandardSequences.Accelerators_cff")
process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")

from HeterogeneousCore.MPICore.modules import *

process.mpiController = MPIController(
    mode = 'CommWorld',
    followerProcessName = 'MPIFollower'
)

# MPIToken has no TrivialSerialisation plugin, so MPISenderPortable falls back
# to ROOT serialisation. But MPIToken is declared "persistent=false" in
# HeterogeneousCore/MPICore/src/classes_def.xml, so ROOT cannot serialise it
# either. With no serialisation mechanism available, MPISenderPortable should
# throw an exception when asked to operate on this product.
process.sender = cms.EDProducer("MPISenderPortable@alpaka",
    upstream = cms.InputTag("mpiController"),
    instance = cms.int32(42),
    products = cms.VPSet(
        cms.PSet(
            type = cms.string("MPIToken"),
            src = cms.InputTag("mpiController"),
        ),
    )
)

process.pathTransient = cms.Path(
    process.mpiController +
    process.sender
)
