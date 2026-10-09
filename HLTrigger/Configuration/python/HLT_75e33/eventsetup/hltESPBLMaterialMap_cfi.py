import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.phase2CAVectorHits_cff import phase2CAVectorHits
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting

# Material map of the BrokenLine fit, for the vector-hits CA
def _addHltESPBLMaterialMap(process):
    process.hltESPBLMaterialMap = cms.ESProducer('BLMaterialMapESProducerAlpaka@alpaka',
        indexFile = cms.FileInPath('RecoTracker/PixelSeeding/data/BLMaterialMap/BLMaterialMap.index'),
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
    )

addHltESPBLMaterialMap = (phase2CAVectorHits | ngtScouting).makeProcessModifier(_addHltESPBLMaterialMap)
