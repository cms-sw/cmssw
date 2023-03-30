import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIServer")

#process.load("FWCore.Services.Tracer_cfi")
process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.MPIService.pmix_server_uri = 'file:server.uri'

process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testfile.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(1),
)

from HeterogeneousCore.MPICore.mpiDriver_cfi import mpiDriver as mpiDriver_
process.mpiDriver = mpiDriver_.clone(
  #eventProducts = [ "things" ]
)
process.path = cms.Path(process.mpiDriver)
