import FWCore.ParameterSet.Config as cms

ParticleFilterBlock = cms.PSet(
    ParticleFilter = cms.PSet(
        # Allow *ALL* protons with energy > protonEMin
        protonEMin = cms.double(5000.0),
        # Particles must have abs(eta) < etaMax (if close enough to 0,0,0)
        etaMax = cms.double(5.3),
        # Charged particles with pT < pTMin (GeV/c) are not simulated
        chargedPtMin = cms.double(0.1),
        # Particles must have energy greater than EMin [GeV]
        EMin = cms.double(0.1),
        # the two following variables define the volume enclosed by the calorimeters
        # radius of the ECAL barrel inner surface
        rMax = cms.double(129.),
        # half-length of the ECAL endcap inner surface
        zMax = cms.double(317.),
        # List of invisible particles (abs of pdgid)
        invisibleParticles = cms.vint32()
        )
    )
