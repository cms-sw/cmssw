import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(35.0),
        MinPt = cms.double(35.0),
        # you can request more than 1 particle
        #untracked vint32  PartID = { 211, 11, -13 }
        PartID = cms.vint32(11),
        MaxEta = cms.double(2.5),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-2.5),
        MinPhi = cms.double(-3.14159265359) ## it must be in radians
    ),
    Verbosity = cms.untracked.int32(0), ## for printouts, set it to 1 (or greater)   

    AddAntiParticle = cms.bool(True), ## back-to-back particles

    firstRun = cms.untracked.uint32(1)
)

ProductionFilterSequence = cms.Sequence(generator)
