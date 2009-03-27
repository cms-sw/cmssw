import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

generator = cms.EDProducer("Pythia6JetGun",
    PGunParameters = cms.PSet(
        ParticleID = cms.vint32(211, -211, 111, 111, 130),
        # this defines "absolute" energy spead of particles in the jet
        MinE = cms.double(0.5),
        MaxE = cms.double(2.0),
        # the following params define the boost
        MinP = cms.double(20.0),
        MaxP = cms.double(20.0),
        MinPhi = cms.double(-3.1415926535),
        MaxPhi = cms.double(+3.1415926535),
        MinEta = cms.double(-2.4),
        MaxEta = cms.double(2.4)
    ),

    # no detailed pythia6 settings necessary    
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring()
    )
)

ProductionFilterSequence = cms.Sequence(generator)
