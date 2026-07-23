import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer(
    "DisplacedParticleGunProducer",
    PGunParameters = cms.PSet(
        # particle direction
        MinPt  = cms.double(5.),
        MaxPt  = cms.double(100.),
        MinPhi = cms.double(-3.141592653589793),
        MaxPhi = cms.double(+3.141592653589793),
        MinTheta = cms.double(-3.141592653589793 / 4),
        MaxTheta = cms.double(3.141592653589793 / 4),

        # displaced vertex in the transverse plane (cm)
        # located in front of HGCAL's CE-E surface (also avoiding the moderator)
        RMin     = cms.double(60.),
        RMax     = cms.double(90.),
        MinVtxPhi = cms.double(0.0),
        MaxVtxPhi = cms.double(2.0 * 3.141592653589793),
        ZVtx     = cms.double(321.),

        NParticles = cms.int32(1),
        PartID     = cms.int32(22), # photon

        # how to sample the vertex radius
        UniformDensityInR = cms.bool(False),
        
        MaxTries = cms.uint32(10000),
        
        # if True: derive theta to hit a region of the HGCAL CE-E back surface
        #  the phi of the particle's direction is matched to the phi of the vertex
        #  which avoid particles crossing the beamline and restricts particles to a tighter region of the detector
        # if False: use MinTheta/MaxTheta
        PointingToHGCAL = cms.bool(True),

        # only used if PointingToHGCAL == True,
        # corresponds to the central third of HGCAL's CE-E back surface
        # note that these values might not be optimal if the default vertex coordinates' configuration is modified
        RMinBackSurfaceHGCAL = cms.double(75.80),
        RMaxBackSurfaceHGCAL = cms.double(120.23),
        
        # if True: the particle's direction is projected back to the z=0 plane,
        #  ensuring compatibility with the specified range
        RestrictRInZPlaneAtZero = cms.bool(True),
        RMinAtZero = cms.double(0.),
        RMaxAtZero = cms.double(100.),

    ),
    Verbosity = cms.untracked.int32(1),
    firstRun  = cms.untracked.uint32(1),
)
