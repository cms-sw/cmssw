import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    comEnergy = cms.double(14000.0),
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
                'Charmonium:states(3S1) = 100443',
                'Charmonium:O(3S1)[3S1(1)] = 0.76',
                'Charmonium:O(3S1)[3S1(8)] = 0.0050',
                'Charmonium:O(3S1)[1S0(8)] = 0.004',
                'Charmonium:O(3S1)[3P0(8)] = 0.004',
                'Charmonium:gg2ccbar(3S1)[3S1(1)]g = on',
                'Charmonium:gg2ccbar(3S1)[3S1(1)]gm = on',
                'Charmonium:gg2ccbar(3S1)[3S1(8)]g = on',
                'Charmonium:qg2ccbar(3S1)[3S1(8)]q = on',
                'Charmonium:qqbar2ccbar(3S1)[3S1(8)]g = on',
                'Charmonium:gg2ccbar(3S1)[1S0(8)]g = on',
                'Charmonium:qg2ccbar(3S1)[1S0(8)]q = on',
                'Charmonium:qqbar2ccbar(3S1)[1S0(8)]g = on',
                'Charmonium:gg2ccbar(3S1)[3PJ(8)]g = on',
                'Charmonium:qg2ccbar(3S1)[3PJ(8)]q = on',
                'Charmonium:qqbar2ccbar(3S1)[3PJ(8)]g = on',
                '100443:onMode = off',
                '100443:onIfMatch = 443 211 -211',
                '443:onMode = off',
                '443:onIfMatch = 13 -13',              
                'PhaseSpace:pTHatMin = 10.'
            ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'processParameters',
                                    )
    )
)

# Filter with high pT cut on dimuon, trying to accomodate trigger requirements.

psi2SIDfilter = cms.EDFilter("PythiaFilter",
    ParticleID = cms.untracked.int32(100443),
    MinPt = cms.untracked.double(0.0),
    MinEta = cms.untracked.double(-2.4),
    MaxEta = cms.untracked.double(2.4),
    Status = cms.untracked.int32(2)
)

jpsifilter = cms.EDFilter("PythiaFilter",
    ParticleID = cms.untracked.int32(443),
    MinPt = cms.untracked.double(0.0),
    MinEta = cms.untracked.double(-2.4),
    MaxEta = cms.untracked.double(2.4),
    Status = cms.untracked.int32(2)
)

# Next two muon filter are derived from muon reconstruction

muminusfilter = cms.EDFilter("PythiaDauVFilter",
    MotherID = cms.untracked.int32(0),
    MinPt = cms.untracked.vdouble(2.),
    ParticleID = cms.untracked.int32(443),
    ChargeConjugation = cms.untracked.bool(False),
    MinEta = cms.untracked.vdouble(-2.4),
    MaxEta = cms.untracked.vdouble(2.4),
    NumberDaughters = cms.untracked.int32(1),
    DaughterIDs = cms.untracked.vint32(-13)
)

muplusfilter = cms.EDFilter("PythiaDauVFilter",
    MotherID = cms.untracked.int32(0),
    MinPt = cms.untracked.vdouble(2.),
    ParticleID = cms.untracked.int32(443),
    ChargeConjugation = cms.untracked.bool(False),
    MinEta = cms.untracked.vdouble(-2.4),
    MaxEta = cms.untracked.vdouble(2.4),
    NumberDaughters = cms.untracked.int32(1),
    DaughterIDs = cms.untracked.vint32(13)
)

#  two pion filter 
piminusfilter = cms.EDFilter("PythiaDauVFilter",
    MotherID = cms.untracked.int32(0),
    MinPt = cms.untracked.vdouble(0.0),
    ParticleID = cms.untracked.int32(100443),
    ChargeConjugation = cms.untracked.bool(False),
    MinEta = cms.untracked.vdouble(-2.4), # or 3.0 ?
    MaxEta = cms.untracked.vdouble(2.4), # or 3.0 ?
    NumberDaughters = cms.untracked.int32(1),
    DaughterIDs = cms.untracked.vint32(-211)
)

piplusfilter = cms.EDFilter("PythiaDauVFilter",
    MotherID = cms.untracked.int32(0),
    MinPt = cms.untracked.vdouble(0.0),
    ParticleID = cms.untracked.int32(100443),
    ChargeConjugation = cms.untracked.bool(False),
    MinEta = cms.untracked.vdouble(-2.4), # or 3.0 ?
    MaxEta = cms.untracked.vdouble(2.4), # or 3.0 ?
    NumberDaughters = cms.untracked.int32(1),
    DaughterIDs = cms.untracked.vint32(211)
)

ProductionFilterSequence = cms.Sequence(generator*psi2SIDfilter*jpsifilter*muminusfilter*muplusfilter*piminusfilter*piplusfilter)
