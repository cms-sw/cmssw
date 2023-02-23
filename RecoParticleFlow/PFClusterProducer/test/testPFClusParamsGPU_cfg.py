import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

###
### EDM Input and job configuration
###
process.source = cms.Source('EmptySource')

# limit the number of events to be processed
process.maxEvents.input = 50

# enable TrigReport, TimeReport and MultiThreading
process.options.wantSummary = True
process.options.numberOfThreads = 4
process.options.numberOfStreams = 0

###
### ESModules, EDModules, Sequences, Tasks, Paths, EndPaths and Schedule
###
process.load('Configuration.StandardSequences.Accelerators_cff')

from RecoParticleFlow.PFClusterProducer.pfClusteringParamsGPUESSource_cfi import pfClusteringParamsGPUESSource as _pfClusteringParamsGPUESSource
process.PFClusteringParamsGPUESSource = _pfClusteringParamsGPUESSource.clone(
  appendToDataLabel = 'pfClusParamsOfflineDefault',
)

from RecoParticleFlow.PFClusterProducer.testDumpPFClusteringParamsGPU_cfi import testDumpPFClusteringParamsGPU as _testDumpPFClusteringParamsGPU
process.theProducer = _testDumpPFClusteringParamsGPU.clone(
  pfClusteringParameters = 'PFClusteringParamsGPUESSource:pfClusParamsOfflineDefault',
)

process.theSequence = cms.Sequence( process.theProducer )

process.thePath = cms.Path( process.theSequence )

process.schedule = cms.Schedule( process.thePath )
