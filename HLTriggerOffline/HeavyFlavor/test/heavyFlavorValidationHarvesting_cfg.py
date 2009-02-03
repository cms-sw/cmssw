import FWCore.ParameterSet.Config as cms

process = cms.Process('HEAVYFLAVORVALIDATIONHARVESTING')

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

process.source = cms.Source("PoolSource",
    processingMode = cms.untracked.string('RunsAndLumis'),
    fileNames = cms.untracked.vstring('file:heavyFlavorValidation.root')
)

process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
process.load('HLTriggerOffline/HeavyFlavor/heavyFlavorValidationHarvestingSequence_cff')

process.path = cms.Path(process.EDMtoME * process.heavyFlavorValidationHarvestingSequence)

process.endpath = cms.EndPath(process.DQMSaver)
