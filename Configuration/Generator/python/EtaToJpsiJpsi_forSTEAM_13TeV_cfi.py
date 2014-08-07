# Only for Trigger Study

import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *

generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(3),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    displayPythiaCards = cms.untracked.bool(False),
    comEnergy = cms.double(13000.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        pythiaEtab = cms.vstring(
          'MSEL=0',
          'MSUB(471)=1',
          'MDME(1521,1)=0',
          'KFDP(1520,1)=443',
          'KFDP(1520,2)=443',
          '443:ALLOFF',
          '443:ONIFMATCH 13 13',
          '333:ALLOFF',
          '333:ONIFMATCH 321 321',
        'PMAS(C10551,1)=7.0'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring(
            'pythiaUESettings',
            'pythiaEtab')
    )
)


etafilter = cms.EDFilter(
        "PythiaFilter",
        MaxEta = cms.untracked.double(9999.),
        MinEta = cms.untracked.double(-9999.),
        ParticleID = cms.untracked.int32(10551)
        )

jpsifilter = cms.EDFilter(
        "PythiaDauVFilter",
        verbose = cms.untracked.int32(0),
        NumberDaughters = cms.untracked.int32(2),
        MotherID = cms.untracked.int32(10551),
        ParticleID = cms.untracked.int32(443),
        DaughterIDs = cms.untracked.vint32(13, -13),
        MinP = cms.untracked.vdouble(2.5,2.5),
        MinEta = cms.untracked.vdouble(-2.6, -2.6),
        MaxEta = cms.untracked.vdouble( 2.6, 2.6)
        )

#mumugenfilter = cms.EDFilter(
# "MCParticlePairFilter",
# Status = cms.untracked.vint32(1, 1),
# MaxEta = cms.untracked.vdouble(2.6, 2.6),
# MinEta = cms.untracked.vdouble(-2.6, -2.6),
# MinP = cms.untracked.vdouble(2.5, 2.5),
# ParticleID1 = cms.untracked.vint32(13),
# ParticleID2 = cms.untracked.vint32(-13)
# )

ProductionFilterSequence = cms.Sequence(generator*etafilter*jpsifilter)
