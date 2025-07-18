import FWCore.ParameterSet.Config as cms

hltPhase2SiPixelClustersSoA = cms.EDProducer("SiPixelPhase2DigiToCluster@alpaka",
    # autoselect the alpaka backend
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
