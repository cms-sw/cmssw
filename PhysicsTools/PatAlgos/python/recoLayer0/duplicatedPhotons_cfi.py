import FWCore.ParameterSet.Config as cms

# Remove duplicates from the photon list

photonsNoDuplicates = cms.EDFilter("DuplicatedPhotonCleaner",
    ## reco photon input source
    photonSource = cms.InputTag("photons"), 

    ## Algorithm used to clean.
    ##   bySeed         = using supercluster seed
    ##   bySuperCluster = using only the supercluster
    removalAlgo  = cms.string("bySeed"),
)
