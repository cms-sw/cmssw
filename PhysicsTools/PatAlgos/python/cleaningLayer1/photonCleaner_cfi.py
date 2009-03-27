import FWCore.ParameterSet.Config as cms

cleanLayer1Photons = cms.EDFilter("PATPhotonCleaner",
    ## Input collection of Photons
    src = cms.InputTag("selectedLayer1Photons"),

    # preselection (any string-based cut for pat::Photon)
    preselection = cms.string(''),

    # overlap checking configurables
    checkOverlaps = cms.PSet(
        electrons = cms.PSet(
           src       = cms.InputTag("cleanLayer1Electrons"),
           algorithm = cms.string("bySuperClusterSeed"),
           requireNoOvelaps = cms.bool(True), # DISCARD photons that overlap!
        ),
    ),

    # finalCut (any string-based cut for pat::Photon)
    finalCut = cms.string(''),

)
