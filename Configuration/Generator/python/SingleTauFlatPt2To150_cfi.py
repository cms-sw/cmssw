import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *

generator = cms.EDFilter("Pythia8PtGun",
                         pythia8CommonSettingsBlock,
                         PGunParameters = cms.PSet(
        ParticleID = cms.vint32(-15),
        AddAntiParticle = cms.bool(False),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinPt = cms.double(2.0),
        MaxPt = cms.double(150.000),
        MinEta = cms.double(-3.1),
        MaxEta = cms.double(3.1)
        ),
                         pythiaTauJets = cms.vstring(
        'ParticleDecays:sophisticatedTau = 2',
        'ParticleDecays:tauPolarization = 0',
        "15:onMode = off",
        "15:onIfAny = 211 -211 321 -321" # turn on if there is a charged k or pi in the decay products 
        ),
                         parameterSets = cms.vstring(
        'pythia8CommonSettings',
        'pythiaTauJets'
        ),
                         PythiaParameters = cms.PSet(parameterSets = cms.vstring())
                         )
