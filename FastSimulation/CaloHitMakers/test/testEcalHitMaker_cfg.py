# The following comments couldn't be translated into the new config version:

# timing and memory checks

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

# Magnetic field full setup
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# Calo geometry service model
process.load("Configuration.Geometry.GeometryIdeal_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# FastCalorimetry
process.load("FastSimulation.Calorimetry.Calorimetry_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    prod = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('TRandom3')
    ),
    saveFileName = cms.untracked.string('')
)

process.source = cms.Source("EmptySource")

from FastSimulation.Calorimetry.Calorimetry_cff import *

process.prod = cms.EDAnalyzer("testEcalHitMaker",
                              FamosCalorimetryBlock,
                              TestParticleFilter = cms.PSet(
    # Particles with |eta| > etaMax (momentum direction at primary vertex) 
    # are not simulated 
    etaMax = cms.double(5.0),
    # Particles with energy smaller than EMin (GeV) are not simulated
    EMin = cms.double(0.0),
    # Allow *ALL* protons with energy > protonEMin
    protonEMin = cms.double(99999.0),
    # Charged particles with pT < pTMin (GeV/c) are not simulated
    chargedPtMin = cms.double(0.0),
    # the two following variables define the volume enclosed by the calorimeters
    # radius of the ECAL barrel inner surface
    rMax = cms.double(129.),
    # half-length of the ECAL endcap inner surface
    zMax = cms.double(317.),
    # List of invisible particles (abs of pdgid)
    invisibleParticles = cms.vint32()
    )
)


process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.prod)


