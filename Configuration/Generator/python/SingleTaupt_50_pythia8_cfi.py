import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
generator = cms.EDFilter("Pythia8PtGun",
                         pythia8CommonSettingsBlock,
                         pythia8CUEP8M1SettingsBlock,
                         PGunParameters = cms.PSet(
        ParticleID = cms.vint32(-15),
        AddAntiParticle = cms.bool(False),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinPt = cms.double(50.0),
        MaxPt = cms.double(50.0001),
        MinEta = cms.double(-2.4),
        MaxEta = cms.double(2.4)
        ),
                         pythiaTauJets = cms.vstring(
        "15:onMode = off",
        "15:onIfAny = 211 -211 321 -321"
        ),
                         parameterSets = cms.vstring(
        'pythia8CommonSettings',
        'pythia8CUEP8M1Settings',
        'pythiaTauJets'
        )
                         )
