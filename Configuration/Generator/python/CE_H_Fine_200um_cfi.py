import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("CloseByParticleGunProducer",
    PGunParameters = cms.PSet(PartID = cms.vint32(22),
        EnMin = cms.double(25.),
        EnMax = cms.double(200.),
        RMin = cms.double(89.99),
        RMax = cms.double(90.01),
        ZMin = cms.double(362.519),
        ZMax = cms.double(362.521),
        Delta = cms.double(10),
        Pointing = cms.bool(True),
        Overlapping = cms.bool(False),
        RandomShoot = cms.bool(False),
        NParticles = cms.int32(1),
        MaxEta = cms.double(2.7),
        MinEta = cms.double(1.7),
        MaxPhi = cms.double(3.14159265359),
        MinPhi = cms.double(-3.14159265359),

    ),
    Verbosity = cms.untracked.int32(0),

    psethack = cms.string('random particles in phi and r windows'),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)
