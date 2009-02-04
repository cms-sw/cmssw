# The following comments couldn't be translated into the new config version:

# module to make electronPixelSeeds via SubSeedGenerator

import FWCore.ParameterSet.Config as cms

siStripSeeds = cms.EDProducer("ElectronSiStripSeedProducer",
    endcapSuperClusters = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters"),
)


