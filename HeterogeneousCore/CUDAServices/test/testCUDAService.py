import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('HeterogeneousCore.CUDAServices.CUDAService_cfi')
process.MessageLogger.categories.append("CUDAService")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 0 )
)
