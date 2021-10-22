import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32( 4 ),
    numberOfStreams = cms.untracked.uint32( 0 ),
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.CUDAService = {}

process.load('HeterogeneousCore.CUDAServices.CUDAService_cfi')
process.CUDAService.verbose = True

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 0 )
)
