# File: Configuration/Generator/python/DisplacedHGCalParticleGun_Photon_cfi.py
import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer(
    "DisplacedHGCalParticleGunProducer",
    PGunParameters = cms.PSet(
        # particle direction
        MinPt  = cms.double(5.0),
        MaxPt  = cms.double(100.0),
        MinPhi = cms.double(-3.141592653589793),
        MaxPhi = cms.double(+3.141592653589793),

        # displaced vertex in transverse plane (cm)
        RMin     = cms.double(0.0),
        RMax     = cms.double(10.0),
        MinVtxPhi = cms.double(0.0),
        MaxVtxPhi = cms.double(2.0 * 3.141592653589793),
        ZVtx     = cms.double(0.0),

        NParticles = cms.int32(1),
        PartID     = cms.vint32(22), # photon

        # how to sample the vertex radius
        UniformDensityInR = cms.bool(False),

        # if True: derive theta to hit a region of the HGCAL front surface
        # if False: use MinTheta/MaxTheta
        PointingToHGCAL = cms.bool(True),

        # only used if PointingToHGCAL == True (cm),
        # corresponds to the central third of HGCAL's front surface
        RminFrontSurfaceHGCAL = cms.double(58.79),
        RmaxFrontSurfaceHGCAL = cms.double(91.58),

        # only used if PointingToHGCAL == False (radians)
        MinTheta = cms.double(0.),
        MaxTheta = cms.double(3.141592653589793),
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun  = cms.untracked.uint32(1),
    psethack  = cms.string("displaced gun with theta, optionally pointing to hard-coded HGCAL front surface"),
)
