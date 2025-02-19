import FWCore.ParameterSet.Config as cms

# configuration of the ExpectedCorrectionsCalculator
#
# we need here the applied misalignments to calculate expected ones
#
ExpectedCorrectionsCalculator = cms.EDFilter("ExpectedAlignmentCorrectionsCalculator",
    # -- Tib- -- //
    TIB2 = cms.PSet(
        BarrelLayer4 = cms.PSet(
            phiZ = cms.double(-0.0004),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        )
    ),
    # -- Tib+ -- //
    TIB1 = cms.PSet(
        BarrelLayer4 = cms.PSet(
            phiZ = cms.double(0.0005),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        )
    ),
    # -- Tob+ -- //
    TOB1 = cms.PSet(
        BarrelLayer1 = cms.PSet(
            phiZ = cms.double(-0.0003),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        )
    ),
    # -- Tob- -- //
    TOB2 = cms.PSet(
        BarrelLayer1 = cms.PSet(
            phiZ = cms.double(-0.0002),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        )
    ),
    # Applied Distortions. Needed To Calculate The Expected Corrections
    # -- Tec+ -- //
    TEC1 = cms.PSet(
        EndcapLayer8 = cms.PSet(
            phiZ = cms.double(0.00019),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer9 = cms.PSet(
            phiZ = cms.double(0.0003),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer1 = cms.PSet(
            phiZ = cms.double(0.0),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer2 = cms.PSet(
            phiZ = cms.double(0.00023),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer3 = cms.PSet(
            phiZ = cms.double(-7e-05),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer4 = cms.PSet(
            phiZ = cms.double(0.00012),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer5 = cms.PSet(
            phiZ = cms.double(0.00018),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer6 = cms.PSet(
            phiZ = cms.double(-0.00014),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer7 = cms.PSet(
            phiZ = cms.double(0.00015),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        )
    ),
    # -- Tec- -- //
    TEC2 = cms.PSet(
        EndcapLayer8 = cms.PSet(
            phiZ = cms.double(-0.00019),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer9 = cms.PSet(
            phiZ = cms.double(0.0003),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer1 = cms.PSet(
            phiZ = cms.double(0.0),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer2 = cms.PSet(
            phiZ = cms.double(0.00033),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer3 = cms.PSet(
            phiZ = cms.double(-0.00011),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer4 = cms.PSet(
            phiZ = cms.double(0.00014),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer5 = cms.PSet(
            phiZ = cms.double(0.00021),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer6 = cms.PSet(
            phiZ = cms.double(-0.00014),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer7 = cms.PSet(
            phiZ = cms.double(5e-05),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        )
    )
)


