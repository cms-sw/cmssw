import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIFollower")

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
process.options.wantSummary = False

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")

from HeterogeneousCore.MPICore.modules import *

process.source = MPISource()

process.maxEvents.input = -1

# receive and validate a portable object, a portable collection, and some portable multi-block collections
process.receiver = MPIReceiver(
    upstream = "source",
    instance = 42,
    products = [
        dict(
            type = "PortableHostObject<portabletest::TestStruct>",
            label = ""
        ),
        dict(
            type = "PortableHostCollection<portabletest::TestSoALayout<128,false> >",
            label = ""
        ),
        dict(
            type = "PortableHostCollection<portabletest::SoABlocks2<128,false> >",
            label = ""
        ),
        dict(
            type = "PortableHostCollection<portabletest::SoABlocks3<128,false> >",
            label = ""
        ),
        dict(
            type = "ushort",
            label = "backend"
        )
    ]
)

process.validatePortableCollections = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("receiver")
)

process.validatePortableObject = cms.EDAnalyzer("TestAlpakaObjectAnalyzer",
    source = cms.InputTag("receiver")
)

process.pathSoA = cms.Path(
    process.receiver +
    process.validatePortableCollections +
    process.validatePortableObject
)
