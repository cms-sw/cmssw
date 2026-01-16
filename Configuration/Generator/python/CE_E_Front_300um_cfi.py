import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("CloseByParticleGunProducer",
    PGunParameters = cms.PSet(PartID = cms.vint32(22),
        ControlledByREta = cms.bool(False),
        ControlledByEta = cms.bool(False),
        VarMin = cms.double(25.),
        VarMax = cms.double(200.),
        MaxVarSpread = cms.bool(False),
        FlatPtGeneration = cms.bool(False),
        RMin = cms.double(134.99),
        RMax = cms.double(135.01),
        ZMin = cms.double(320.99),
        ZMax = cms.double(321.01),
        Delta = cms.double(10),
        Pointing = cms.bool(True),
        Overlapping = cms.bool(False),
        RandomShoot = cms.bool(False),
        NParticles = cms.int32(1),
        MaxEta = cms.double(2.7),
        MinEta = cms.double(1.7),
        MaxPhi = cms.double(3.14159265359),
        MinPhi = cms.double(-3.14159265359),
        UseDeltaT = cms.bool(False),
        TMin = cms.double(0.),
        TMax = cms.double(0.05),
        OffsetFirst = cms.double(0.)
    ),
    Verbosity = cms.untracked.int32(0),

    psethack = cms.string('random particles in phi and r windows'),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)
