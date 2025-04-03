import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *

generator = cms.EDFilter("Pythia8PtExpGun",

    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    
    PGunParameters = cms.PSet(
        ParticleID = cms.vint32(221),
        AddAntiParticle = cms.bool(False),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        #MinPt = cms.double(5.0),
        MinPt = cms.double(10.0),
        #MaxPt = cms.double(65.0),
        #MaxPt = cms.double(25.0),
        MaxPt = cms.double(65.0),
        MinEta = cms.double(-2.4),
        MaxEta = cms.double(2.4)
        ),
    
    PythiaParameters = cms.PSet(
            pythia8CommonSettingsBlock,
            pythia8CP5SettingsBlock,
            processParameters = cms.vstring(
                #'SLHA:keepSM = on',
                #'SLHA:minMassSM = 10.',
                # Very important to enable override!
                'SLHA:allowUserOverride = on',
                'RHadrons:allow = on',
                'RHadrons:allowDecay = on',
                #'32:mayDecay = true',
                '221:mayDecay = true',
                # Set decay channels of eta (mumugamma)
                '221:oneChannel = 1 1.0 0 13 -13 22'
                ),
            parameterSets = cms.vstring(
                'pythia8CommonSettings',
                'pythia8CP5Settings',
                'processParameters',
                )
    )
)
