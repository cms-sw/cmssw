import FWCore.ParameterSet.Config as cms

# Material map of the BrokenLine fit
hltESPBLMaterialMap = cms.ESProducer('BLMaterialMapESProducerAlpaka@alpaka',
        indexFile = cms.FileInPath('RecoTracker/PixelSeeding/data/BLMaterialMap/BLMaterialMap.index'),
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
