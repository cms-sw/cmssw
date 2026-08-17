import FWCore.ParameterSet.Config as cms

# BL-fit Geant4 material map rho(r,z) as a device-resident EventSetup condition,
# consumed by the Phase-2 pixel CA / BrokenLine fit.
hltESPBLMaterialMap = cms.ESProducer('BLMaterialMapESProducerAlpaka@alpaka',
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
