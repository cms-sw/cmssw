# The following comments couldn't be translated into the new config version:

# timing and memory checks

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

# Magnetic field full setup
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# Calo geometry service model
process.load("Configuration.StandardSequences.Geometry_cff")

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
    # Charged particles with pT < pTMin (GeV/c) are not simulated
    pTMin = cms.double(0.0),
    # Particles with energy smaller than EMin (GeV) are not simulated
    EMin = cms.double(0.0),
    # Protons with energy in excess of this value (GeV) will kept no matter what
    EProton = cms.double(99999.0)
    )
)


process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.prod)


