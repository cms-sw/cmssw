
import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomPtGunProducer",
                           PGunParameters = cms.PSet(
        PartID = cms.vint32(15),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359), ## in radians
        MinEta = cms.double(-2.0),
        MaxEta = cms.double(2.0),
        MinPt = cms.double(1.), # in GeV
        MaxPt = cms.double(20.0)
        ),
                           Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
                           AddAntiParticle = cms.bool(True),
                           )
