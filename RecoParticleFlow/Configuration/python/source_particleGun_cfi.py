import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *
source = cms.Source("PythiaSource",
    Phimin = cms.untracked.double(0.0),
    #  possibility to run single or double back-to-back particles with PYTHIA
    # if ParticleID = 0, run PYTHIA
    ParticleID = cms.untracked.int32(11),
    Etamin = cms.untracked.double(-3.0),
    DoubleParticle = cms.untracked.bool(True),
    Phimax = cms.untracked.double(360.0),
    Ptmin = cms.untracked.double(0.0),
    Ptmax = cms.untracked.double(100.0001),
    Etamax = cms.untracked.double(3.0),
    pythiaVerbosity = cms.untracked.bool(False),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings')
    )
)



