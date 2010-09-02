import FWCore.ParameterSet.Config as cms

XtoFFbarFilter = cms.EDFilter("XtoFFbarFilter",
    src = cms.InputTag("genParticles"),

    # Require event to contain X -> f fbar decay, where X and f are specified below.
    # Optionally also require it to contain a Y -> g g-bar decay.

    # Allowed PDG ID codes of mother X particle
    idMotherX = cms.vint32(6000111, 6000112, 6000113),
    # Allowed PDG ID of daughter f-fbar pair (don't specify anti-particle code)
    idDaughterF = cms.vint32(1,2,3,4,5,6,11,13,15), 

    # If the following vectors are empty, it will not be required that a Y --> g g-bar
    # decay is present.
                              
    # Allowed PDG ID codes of mother Y particle
    idMotherY = cms.vint32(6000111, 6000112, 6000113),
    # Allowed PDG ID of daughter g-gbar pair (don't specify anti-particle code)
    idDaughterG = cms.vint32(1,2,3,4,5,6,11,13,15)
)
