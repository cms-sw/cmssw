import FWCore.ParameterSet.Config as cms

# Despite the name of the generator, this generates particles whose
# distribution is flat in log(Pt).
generator = cms.EDProducer("FlatRandomOneOverPtGunProducer",

    PGunParameters = cms.PSet(
        # This specifies range in 1/Pt
        # It coresponds to Pt = 0.1 to 2000 GeV
        MinOneOverPt = cms.double(0.0005),
        MaxOneOverPt = cms.double(10.),
        PartID = cms.vint32(-13),
        MaxEta = cms.double(2.5),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-2.5),
        MinPhi = cms.double(-3.14159265359) ## in radians

    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single mu pt 0.1to1000'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)
