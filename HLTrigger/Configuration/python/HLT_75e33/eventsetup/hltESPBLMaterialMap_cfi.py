import FWCore.ParameterSet.Config as cms

# BL-fit Geant4 material map rho(r,z) as a device-resident EventSetup condition,
# consumed by the Phase-2 pixel CA / BrokenLine fit. The map is selected automatically from the
# ideal-geometry fingerprint against RecoTracker/PixelSeeding/data/BLMaterialMap/BLMaterialMap.index
# (one file per tracker geometry); a geometry without a map is a hard failure.
hltESPBLMaterialMap = cms.ESProducer('BLMaterialMapESProducerAlpaka@alpaka',
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
