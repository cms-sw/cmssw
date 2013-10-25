import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

generator = cms.EDProducer("FlatRandomPtGunProducer",
    firstRun = cms.untracked.uint32(1),
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        # you can request more than 1 particle
        # PartID = cms.vint32(211,11,-13),
        MinPt = cms.double(10.0),
        MaxPt = cms.double(10.0),
        MinEta = cms.double(-3.0),
        MaxEta = cms.double(3.0),
        MinPhi = cms.double(-3.14159265359), ## it must be in radians
        MaxPhi = cms.double(3.14159265359),
    ),
    AddAntiParticle = cms.bool(False), # back-to-back particles
    Verbosity = cms.untracked.int32(0) ## for printouts, set it to 1 (or greater)   
)

ProductionFilterSequence = cms.Sequence(generator)
