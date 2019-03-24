import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("CloseByParticleGunProducer",
    PGunParameters = cms.PSet(PartID = cms.vint32(22),
        EnMin = cms.double(25.),
        EnMax = cms.double(200.),
        RMin = cms.double(60),
        RMax = cms.double(120),
        ZMin = cms.double(320),
        ZMax = cms.double(650),
        Delta = cms.double(2.5),
        Pointing = cms.bool(True),
        MaxEta = cms.double(2.7),
        MinEta = cms.double(1.7),
        MaxPhi = cms.double(3.14159265359/6.),
        MinPhi = cms.double(-3.14159265359/6.),
                          
    ),
    Verbosity = cms.untracked.int32(10),
    
    psethack = cms.string('single gamma random energy'),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)
