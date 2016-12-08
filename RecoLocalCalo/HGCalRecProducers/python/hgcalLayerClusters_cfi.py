import FWCore.ParameterSet.Config as cms

#### PF CLUSTER ECAL ####

#energy corrector for corrected cluster producer
hgcalLayerClusters =  cms.EDProducer(
    "HGCalClusterTestProducer",
    detector = cms.string("all"),
    doSharing = cms.bool(False),
    deltac = cms.double(2.),
    ecut = cms.double(0.01),
    kappa = cms.double(10.),
    multiclusterRadius = cms.double(0.015),
    verbosity = cms.untracked.uint32(3)
    )

