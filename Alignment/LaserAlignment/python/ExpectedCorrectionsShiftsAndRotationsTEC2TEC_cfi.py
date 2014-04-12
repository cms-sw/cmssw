import FWCore.ParameterSet.Config as cms

# configuration of the ExpectedCorrectionsCalculator
#
# we need here the applied misalignments to calculate expected ones
#
ExpectedCorrectionsCalculator = cms.EDFilter("ExpectedAlignmentCorrectionsCalculator",
    # -- TIB- -- //
    TIB2 = cms.PSet(
        BarrelLayer4 = cms.PSet(
            phiZ = cms.double(-0.0004),
            dX = cms.double(-0.0089),
            dY = cms.double(0.034)
        )
    ),
    # -- TIB+ -- //
    TIB1 = cms.PSet(
        BarrelLayer4 = cms.PSet(
            phiZ = cms.double(0.0005),
            dX = cms.double(0.0208),
            dY = cms.double(0.0047)
        )
    ),
    # -- TOB+ -- //
    TOB1 = cms.PSet(
        BarrelLayer1 = cms.PSet(
            phiZ = cms.double(-0.0003),
            dX = cms.double(-0.0125),
            dY = cms.double(-0.0095)
        )
    ),
    # -- TOB- -- //
    TOB2 = cms.PSet(
        BarrelLayer1 = cms.PSet(
            phiZ = cms.double(-0.0002),
            dX = cms.double(0.0025),
            dY = cms.double(0.034)
        )
    ),
    # applied distortions. Needed to calculate the expected corrections
    # -- TEC+ -- //
    TEC1 = cms.PSet(
        EndcapLayer8 = cms.PSet(
            phiZ = cms.double(0.00019),
            dX = cms.double(0.0033),
            dY = cms.double(-0.013)
        ),
        EndcapLayer9 = cms.PSet(
            phiZ = cms.double(0.0003),
            dX = cms.double(-0.011),
            dY = cms.double(0.0031)
        ),
        EndcapLayer1 = cms.PSet(
            phiZ = cms.double(0.00018),
            dX = cms.double(0.0009),
            dY = cms.double(-0.015)
        ),
        EndcapLayer2 = cms.PSet(
            phiZ = cms.double(0.00023),
            dX = cms.double(0.003),
            dY = cms.double(-0.0025)
        ),
        EndcapLayer3 = cms.PSet(
            phiZ = cms.double(-7e-05),
            dX = cms.double(-0.002),
            dY = cms.double(0.0017)
        ),
        EndcapLayer4 = cms.PSet(
            phiZ = cms.double(0.00012),
            dX = cms.double(0.01),
            dY = cms.double(0.013)
        ),
        EndcapLayer5 = cms.PSet(
            phiZ = cms.double(0.0),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer6 = cms.PSet(
            phiZ = cms.double(-0.00014),
            dX = cms.double(-0.0015),
            dY = cms.double(0.0023)
        ),
        EndcapLayer7 = cms.PSet(
            phiZ = cms.double(0.00015),
            dX = cms.double(0.0022),
            dY = cms.double(-0.0012)
        )
    ),
    # -- TEC- -- //
    TEC2 = cms.PSet(
        EndcapLayer8 = cms.PSet(
            phiZ = cms.double(-0.00019),
            dX = cms.double(0.0033),
            dY = cms.double(-0.0013)
        ),
        EndcapLayer9 = cms.PSet(
            phiZ = cms.double(0.0003),
            dX = cms.double(-0.011),
            dY = cms.double(0.031)
        ),
        EndcapLayer1 = cms.PSet(
            phiZ = cms.double(-0.00026),
            dX = cms.double(0.0),
            dY = cms.double(0.0)
        ),
        EndcapLayer2 = cms.PSet(
            phiZ = cms.double(0.00033),
            dX = cms.double(0.003),
            dY = cms.double(-0.0025)
        ),
        EndcapLayer3 = cms.PSet(
            phiZ = cms.double(-0.00011),
            dX = cms.double(-0.02),
            dY = cms.double(0.017)
        ),
        EndcapLayer4 = cms.PSet(
            phiZ = cms.double(0.00014),
            dX = cms.double(0.001),
            dY = cms.double(0.013)
        ),
        EndcapLayer5 = cms.PSet(
            phiZ = cms.double(0.00021),
            dX = cms.double(0.009),
            dY = cms.double(-0.015)
        ),
        EndcapLayer6 = cms.PSet(
            phiZ = cms.double(-0.00014),
            dX = cms.double(-0.015),
            dY = cms.double(0.0023)
        ),
        EndcapLayer7 = cms.PSet(
            phiZ = cms.double(5e-05),
            dX = cms.double(0.022),
            dY = cms.double(-0.0007)
        )
    )
)


