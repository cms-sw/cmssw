import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIFollower")

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
process.options.wantSummary = False

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")

process.load("Configuration.StandardSequences.Accelerators_cff")
process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")

from HeterogeneousCore.MPICore.modules import *

process.source = MPISource(mode = 'CommWorld',
    controllerProcessName = 'MPIController'
)

process.maxEvents.input = -1

# MPIToken has no TrivialSerialisation plugin, so MPIReceiverPortable falls back
# to ROOT serialisation. But MPIToken is declared "persistent=false" in
# HeterogeneousCore/MPICore/src/classes_def.xml, so ROOT cannot serialise it
# either. With no serialisation mechanism available, MPIReceiverPortable should
# throw an exception when asked to operate on this product.
process.receiver = cms.EDProducer("MPIReceiverPortable@alpaka",
    upstream = cms.InputTag("source"),
    instance = cms.int32(42),
    products = cms.VPSet(
        cms.PSet(
            type = cms.string("MPIToken"),
            src = cms.InputTag("", ""),
        ),
    )
)

process.pathTransient = cms.Path(
    process.receiver
)
