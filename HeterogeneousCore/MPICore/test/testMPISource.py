import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIClient")

#process.load("FWCore.Services.Tracer_cfi")
process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.MPIService.pmix_server_uri = 'file:server.uri'

process.source = cms.Source("MPISource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(1),
)

# FIXME this module needs to be updated to a different data product
#from HeterogeneousCore.MPICore.mpiReporter_cfi import mpiReporter as mpiReporter_
#process.mpiReporter = mpiReporter_.clone()
#process.path = cms.Path(process.mpiReporter)
