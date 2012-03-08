import FWCore.ParameterSet.Config as cms
from Configuration.Generator.PythiaUEZ2starSettings_cfi import *

generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(0.002305),
    comEnergy = cms.double(8000.0),
    crossSection = cms.untracked.double(51560000000.),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring(
            'MSEL=1           ! User defined processes',
            'MSTJ(22)=4       ! Decay unstable particles in a cylinder',
            'PARJ(73)=1500.   ! max. radius for MSTJ(22)=4',
            'PARJ(74)=3000.   ! max. Z for MSTJ(22)=4',
            'MDCY(C130,1)=1   ! decay k0-longs',
            'MDCY(C211,1)=1   ! decay pions',
            'MDCY(C321,1)=1   ! decay kaons'),

        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)

mugenfilter = cms.EDFilter("MCSmartSingleParticleFilter",
                           MinPt = cms.untracked.vdouble(2.5,2.5),
                           MinEta = cms.untracked.vdouble(-2.5,-2.5),
                           MaxEta = cms.untracked.vdouble(2.5,2.5),
                           ParticleID = cms.untracked.vint32(13,-13),
                           Status = cms.untracked.vint32(1,1),
                           # Decay cuts are in mm
                           MaxDecayRadius = cms.untracked.vdouble(1500.,1500.),
                           MinDecayZ = cms.untracked.vdouble(-3000.,-3000.),
                           MaxDecayZ = cms.untracked.vdouble(3000.,3000.)
)

ProductionFilterSequence = cms.Sequence(generator*mugenfilter)
