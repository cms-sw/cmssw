import FWCore.ParameterSet.Config as cms

cleanPatTracks = cms.EDProducer("PATGenericParticleCleaner",
    src = cms.InputTag("REPLACE_ME"), 

    # preselection (any string-based cut on pat::GenericParticle)
    preselection = cms.string(''),

    # overlap checking configurables
    checkOverlaps = cms.PSet(
        muons = cms.PSet(
           src       = cms.InputTag("cleanPatMuons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),
           deltaR              = cms.double(0.3),
           checkRecoComponents = cms.bool(True), # remove them if the use the same reco::Track
           pairCut             = cms.string(""),
           requireNoOverlaps   = cms.bool(True), # overlaps don't cause the electron to be discared
        ),
        electrons = cms.PSet(
           src       = cms.InputTag("cleanPatElectrons"),
           algorithm = cms.string("byDeltaR"),
           preselection        = cms.string(""),
           deltaR              = cms.double(0.3),
           checkRecoComponents = cms.bool(False), # don't check if they share some AOD object ref
                                                  # as electrons have reco::GsfTrack, not reco::Track
           pairCut             = cms.string("0.5 < ele.pt/part.pt < 1.5"),  # let's do a check on relative P_T
                                                                            # 'part' is the generic particle, 'ele' the electron
           requireNoOverlaps   = cms.bool(True), # overlaps don't cause the electron to be discared
        ),
    ),

    # finalCut (any string-based cut on pat::GenericParticle)
    finalCut = cms.string(''),
)
