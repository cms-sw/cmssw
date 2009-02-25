import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

generator = cms.EDProducer("PythiaProducer",
    pythiaFrame = cms.string("CMS"),
    comEnergy = cms.double(14000.),
    pythiaVerbosity = cms.untracked.bool(False),
    # Generate a number of particles with some fermi motion
    ParticleIDs = cms.vint32(211,-211,111,111,130),
    Emin = cms.untracked.double(0.5),
    Emax = cms.untracked.double(2.0),
    # Then boost them
    #   (Ptmin/Ptmax is the boost absolute value range)
    Pmin = cms.untracked.double(20.0),
    Pmax = cms.untracked.double(20.0),
    #   (Etamin/Etamax is the boost eta range)
    MinEta = cms.untracked.double(-2.4),
    MaxEta = cms.untracked.double(2.4),
    #   (Phimin/Phimax is the boost phi range)
    MinPhi = cms.untracked.double(-3.1415926535),
    MaxPhi = cms.untracked.double(+3.1415926535),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring()
    )

)

ProductionFilterSequence = cms.Sequence(generator)
