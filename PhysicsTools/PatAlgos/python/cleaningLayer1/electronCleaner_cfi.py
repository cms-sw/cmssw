import FWCore.ParameterSet.Config as cms

cleanLayer1Electrons = cms.EDFilter("PATElectronCleaner",
    ## pat electron input source
    src = cms.InputTag("selectedLayer1Electrons"), 

    # preselection (any string-based cut for pat::Electron)
    preselection = cms.string(''),

    # overlap checking configurables
    checkOverlaps = cms.PSet(
        muons = cms.PSet(
           src       = cms.InputTag("cleanLayer1Muons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),  # don't preselect the muons
           deltaR              = cms.double(0.3),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOvelaps = cms.bool(False), # overlaps don't cause the electron to be discared
        )
    ),

    # finalCut (any string-based cut for pat::Electron)
    finalCut = cms.string(''),
)
