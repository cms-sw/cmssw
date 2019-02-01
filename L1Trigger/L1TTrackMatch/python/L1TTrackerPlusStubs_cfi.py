import FWCore.ParameterSet.Config as cms
import math
from math import pi

l1StubMatchedMuons = cms.EDProducer("L1TTrackerPlusStubsProducer",
    srcStubs = cms.InputTag("simKBmtfStubs"),
    srcTracks = cms.InputTag("TTTracksFromTracklet:Level1TTTracks"),
    maxChi2 = cms.double(100),
    trackMatcherSettings = cms.PSet(
        sectorsToProcess = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11),
	verbose = cms.int32(0),
        geomPi = cms.double(pi),
        propagationConstants = cms.vdouble(-1.14441e0,-1.24939e0,-1.31598e0,-1.34792e0),
	sectorSettings = cms.PSet(
            verbose = cms.int32(0),
            geomPi = cms.double(pi),
	    stationsToProcess = cms.vint32(1,2,3,4),
	    tolerance = cms.double(3),
	    toleranceB = cms.double(3),
	    toleranceQ = cms.int32(2),
	    phi1 = cms.vdouble(11*pi/12,-11*pi/12,-9*pi/12,-7*pi/12,-5*pi/12,-3*pi/12,-pi/12,pi/12,3*pi/12,5*pi/12,7*pi/12,9*pi/12),
	    phi2 = cms.vdouble(-11*pi/12,-9*pi/12,-7*pi/12,-5*pi/12,-3*pi/12,-pi/12,pi/12,3*pi/12,5*pi/12,7*pi/12,9*pi/12,11*pi/12),
            propagationConstants = cms.vdouble(-1.14441e0,-1.24939e0,-1.31598e0,-1.34792e0),
	    propagationConstantsB = cms.vdouble(-9.05378e-2,-6.34508e-2,-3.22425e-2,-9.16026e-3),
	    etaHigh2 = cms.vdouble(1.25,1.1,0.95,0.835),
	    etaHigh1 = cms.vdouble(0.85,0.725,0.615,0.54),
	    etaHigh0 = cms.vdouble(0.3,0.25,0.21,0.175),
	    etaHighm1 = cms.vdouble(-0.325,-0.25,-0.225,-0.19),
	    etaHighm2 = cms.vdouble(-0.815,-0.72,-0.6,-0.525),
	    etaLow2 = cms.vdouble(0.815,0.72,0.6,0.525),
	    etaLow1 = cms.vdouble(0.325,0.25,0.225,0.19),
	    etaLow0 = cms.vdouble(-0.3,-0.25,-0.21,-0.175),
	    etaLowm1 = cms.vdouble(-0.85,-0.725,-0.615,-0.54),
	    etaLowm2 = cms.vdouble(-1.25,-1.1,-0.95,-0.835),
	    alpha = cms.vdouble(2.50016e-2,3.10230e-2,3.65267e-2,3.46650e-2),
	    beta = cms.vdouble(2*2.13888e1,2*2.15320e1,2*2.09371e1,2*2.98381e1),
	    alphaB = cms.vdouble(5.56536e-3,6.74712e-3,8.22339e-3,1.04466e-2),
	    betaB = cms.vdouble(1.68262e0,1.61702e0,1.52894e0,1.25928e0)
        )
        
    )
)
