import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer(
    "DisplacedParticleGunProducer",
    PGunParameters = cms.PSet(
        # particle direction
        MinPt  = cms.double(5.0),
        MaxPt  = cms.double(100.0),
        MinPhi = cms.double(-3.141592653589793),
        MaxPhi = cms.double(+3.141592653589793),

        # displaced vertex in the transverse plane (cm)
        # located in front of HGCAL's CE-E surface (also avoiding the moderator)
        RMin     = cms.double(50.),
        RMax     = cms.double(130.),
        MinVtxPhi = cms.double(0.0),
        MaxVtxPhi = cms.double(2.0 * 3.141592653589793),
        ZVtx     = cms.double(321.),

        NParticles = cms.int32(1),
        PartID     = cms.int32(22), # photon

        # how to sample the vertex radius
        UniformDensityInR = cms.bool(False),

        # if True: derive theta to hit a region of the HGCAL front surface
        # if False: use MinTheta/MaxTheta
        PointingToHGCAL = cms.bool(True),

        # only used if PointingToHGCAL == True (cm),
        # corresponds to the central third of HGCAL's CE-E back surface
        # note that these values might not be optimal if the default vertex coordinates are modified
        RminBackSurfaceHGCAL = cms.double(75.80),
        RmaxBackSurfaceHGCAL = cms.double(120.23),

        # only used if PointingToHGCAL == False (radians)
        MinTheta = cms.double(1E-6),
        MaxTheta = cms.double(3.141592653589793),
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun  = cms.untracked.uint32(1),
    psethack  = cms.string("displaced gun with theta, optionally pointing to hard-coded HGCAL front surface"),
)
