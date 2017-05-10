import FWCore.ParameterSet.Config as cms

XtoFFbarFilter = cms.EDFilter("XtoFFbarFilter",
    src = cms.InputTag("genParticles"),

    # Require event to contain decay of X -> f f'-bar + anything.
    # Optionally also require it to contain decay Y -> g g'-bar + anything.
    # N.B. Particles f and f' must both appear in idDaughterF list, but need not be identical to each other.
    # N.B. Particles g and g' must both appear in idDaughterG list, but need not be identical to each other.

    # Allowed PDG ID codes of mother X particle
    idMotherX = cms.vint32(6000111, 6000112, 6000113),
    # Allowed PDG ID of daughter f or f'
    idDaughterF = cms.vint32(1,2,3,4,5,6,11,13,15), 

    # If the following vectors are empty, it will not be required that decay Y --> g g'-bar + anything
    # is present.
                              
    # Allowed PDG ID codes of mother Y particle
    idMotherY = cms.vint32(6000111, 6000112, 6000113),
    # Allowed PDG ID of daughter g or g'
    idDaughterG = cms.vint32(1,2,3,4,5,6,11,13,15),

    # If this is set true, then parameter idMotherY is ignored, and instead set equal to idMotherX.
    # Furthermore, events are vetoed if they contain more than one species from the list idMotherX. 
    idYequalsX = cms.bool(False)
)
