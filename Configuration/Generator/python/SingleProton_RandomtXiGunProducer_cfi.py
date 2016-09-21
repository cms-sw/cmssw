import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("RandomtXiGunProducer",
           PGunParameters = cms.PSet(
                            PartID = cms.vint32(2212),
                            MinPhi = cms.double(-3.14159265359),
                            MaxPhi = cms.double(-3.14159265359),
                            ECMS   = cms.double(13000),
                            Mint   = cms.double(0.),
                            Maxt   = cms.double(2.0),
                            MinXi  = cms.double(0.01),
                            MaxXi  = cms.double(0.2)
           ),
           Verbosity = cms.untracked.int32(0),
           psethack = cms.string('single protons'),
           FireBackward = cms.bool(True),
           FireForward  = cms.bool(True),
           firstRun = cms.untracked.uint32(1)
)
