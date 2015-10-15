import FWCore.ParameterSet.Config as cms

ParticleFilterBlock = cms.PSet(
    ParticleFilter = cms.PSet(
        # Protons with energy larger than EProtons (GeV) are all kept
        EProton = cms.double(5000.0),
        # Particles with |eta| > etaMax (momentum direction at primary vertex) 
        # are not simulated 
        etaMax = cms.double(5.3),
        # Charged particles with pT < pTMin (GeV/c) are not simulated
        pTMin = cms.double(0.1),
        # Particles with energy smaller than EMin (GeV) are not simulated
        EMin = cms.double(0.1)
    )
)

