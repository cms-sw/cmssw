import FWCore.ParameterSet.Config as cms

cleanLayer1Jets = cms.EDFilter("PATJetCleaner",
    src = cms.InputTag("selectedLayer1Jets"), 

    # preselection (any string-based cut on pat::Jet)
    preselection = cms.string(''),

    # overlap checking configurables
    checkOverlaps = cms.PSet(
        muons = cms.PSet(
           src       = cms.InputTag("cleanLayer1Muons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),
           deltaR              = cms.double(0.5),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOvelaps = cms.bool(False), # overlaps don't cause the electron to be discared
        ),
        electrons = cms.PSet(
           src       = cms.InputTag("cleanLayer1Electrons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),
           deltaR              = cms.double(0.5),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOvelaps = cms.bool(False), # overlaps don't cause the electron to be discared
        ),
        photons = cms.PSet(
           src       = cms.InputTag("cleanLayer1Photons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),
           deltaR              = cms.double(0.5),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOvelaps = cms.bool(False), # overlaps don't cause the electron to be discared
        ),
        taus = cms.PSet(
           src       = cms.InputTag("cleanLayer1Taus"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),
           deltaR              = cms.double(0.5),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOvelaps = cms.bool(False), # overlaps don't cause the electron to be discared
        ),
        tkIsoElectrons = cms.PSet(
           src       = cms.InputTag("cleanLayer1Electrons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string("pt > 10 && trackIso < 3"),
           deltaR              = cms.double(0.3),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
           pairCut             = cms.string(""),
           requireNoOvelaps = cms.bool(False), # overlaps don't cause the electron to be discared
        )
    ),


    # finalCut (any string-based cut on pat::Jet)
    finalCut = cms.string(''),
)
