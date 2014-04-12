import FWCore.ParameterSet.Config as cms

MuScleFitGenFilter = cms.EDFilter(
    "MuScleFitGenFilter",

    GenParticlesName = cms.untracked.string( "genParticles" ),

    # The resonances are to be specified in this order:
    # Z0, Y(3S), Y(2S), Y(1S), Psi(2S), J/Psi
    # -------------------------------------------------
    resfind = cms.vint32(0, 0, 0, 0, 0, 1)
)
