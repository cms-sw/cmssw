import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(5.),
        MinPt = cms.untracked.double(100.),
        PartID = cms.untracked.vint32(11),
        MaxEta = cms.untracked.double(2.5),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-2.5),
        MinPhi = cms.untracked.double(-3.14159265359) ## in radians

    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single electron pt 5 to 100'),
    AddAntiParticle = cms.untracked.bool(True),
    firstRun = cms.untracked.uint32(1)
)



ProductionFilterSequence = cms.Sequence(generator)
