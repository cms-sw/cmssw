import FWCore.ParameterSet.Config as cms

tauHadronDecayFilter = cms.EDFilter("TauHadronDecayFilter",
    # using FastSimulation/Event
    particles = cms.InputTag("particleFlowBlock"),
    ParticleFilter = cms.PSet(
        # Particles with |eta| > etaMax (momentum direction at primary vertex)
        # are not simulated
        etaMax = cms.double(10.0),
        # Charged particles with pT < pTMin (GeV/c) are not simulated
        pTMin = cms.double(0.0),
        # Particles with energy smaller than EMin (GeV) are not simulated
        EMin = cms.double(0.0)
    )
)


