import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

from Configuration.Generator.PythiaUESettings_cfi import *
generator = cms.EDProducer("Pythia6PtGun",
    VertexSmearing = cms.PSet(refToPSet_ = cms.string("VertexSmearingParameters")),
    PGunParameters = cms.PSet(
       ParticleID = cms.vint32(11),
       MinPhi = cms.double(0.0),
       MaxPhi = cms.double(360.0),
       MinEta = cms.double(-3.0),
       MaxEta = cms.double(3.0),
       MinPt = cms.double(0.0),
       MaxPt = cms.double(100.0001),
       AddAntiParticle = cms.bool(False)
    ),
    pythiaVerbosity = cms.untracked.bool(False),
    PythiaParameters = cms.PSet(
       pythiaUESettingsBlock,
    # This is a vector of ParameterSet names to be read, in this order
       parameterSets = cms.vstring('pythiaUESettings')
    )
)

ProductionFilterSequence = cms.Sequence(generator)
