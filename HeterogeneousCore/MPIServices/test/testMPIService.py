import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('HeterogeneousCore.MPIServices.MPIService_cfi')
process.MessageLogger.categories.append("MPIService")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 0 )
)
