import FWCore.ParameterSet.Config as cms

cleanPatPhotons = cms.EDProducer("PATPhotonCleaner",
    ## Input collection of Photons
    src = cms.InputTag("selectedPatPhotons"),

    # preselection (any string-based cut for pat::Photon)
    preselection = cms.string(''),

    # overlap checking configurables
    checkOverlaps = cms.PSet(
        electrons = cms.PSet(
           src       = cms.InputTag("cleanPatElectrons"),
           algorithm = cms.string("bySuperClusterSeed"),
           requireNoOverlaps = cms.bool(False), # mark photons that overlap with electrons
                                                # for further studies, but DO NOT discard
                                                # them
        ),
    ),

    # finalCut (any string-based cut for pat::Photon)
    finalCut = cms.string(''),

)
