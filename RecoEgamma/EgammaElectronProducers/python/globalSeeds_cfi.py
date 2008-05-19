# The following comments couldn't be translated into the new config version:

# module to make electronPixelSeeds via SubSeedGenerator

import FWCore.ParameterSet.Config as cms

electronPixelSeedsForGlobalGsfElectrons = cms.EDProducer("GlobalSeedProducer",
    endcapSuperClusters = cms.InputTag("correctedIslandEndcapSuperClusters"),
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters"),
    seedPt = cms.double(0.0),
    seedDPhi = cms.double(0.1),
    seedDr = cms.double(0.3),
    seedDEta = cms.double(0.025),
    initialSeeds = cms.InputTag("hltL1IsoEgammaRegionalPixelSeedGenerator")
)


