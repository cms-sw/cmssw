import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

from Configuration.Generator.PythiaUESettings_cfi import *
generator = cms.EDProducer("Pythia6PtGun",
    pythiaVerbosity = cms.untracked.bool(False),
    PGunParameters = cms.PSet(
        ParticleID = cms.vint32(1),
        AddAntiParticle = cms.bool(True),
        MinPt = cms.double(20.0),
        MaxPt = cms.double(700.0),
#        MinE = cms.double(10.0),
#        MaxE = cms.double(10.0),
        MinEta = cms.double(-1.0),
        MaxEta = cms.double(1.0),
        MinPhi = cms.double(-3.1415926535897931),
        MaxPhi = cms.double(3.1415926535897931)
    ),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        # Tau jets only
        pythiaJets = cms.vstring(),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring(
            'pythiaUESettings',
            'pythiaJets'
        )
    )
)

ProductionFilterSequence = cms.Sequence(generator)
