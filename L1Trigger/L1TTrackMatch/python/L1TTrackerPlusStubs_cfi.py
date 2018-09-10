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
        sectorSettings = cms.PSet(
            verbose = cms.int32(0),
            stationsToProcess = cms.vint32(1,2,3,4),
	    tolerance = cms.int32(3),
	    phi1 = cms.vdouble(11*pi/12,-11*pi/12,-9*pi/12,-7*pi/12,-5*pi/12,-3*pi/12,-pi/12,pi/12,3*pi/12,5*pi/12,7*pi/12,9*pi/12),
	    phi2 = cms.vdouble(-11*pi/12,-9*pi/12,-7*pi/12,-5*pi/12,-3*pi/12,-pi/12,pi/12,3*pi/12,5*pi/12,7*pi/12,9*pi/12,11*pi/12),
            propagationConstants  = cms.vdouble(1.14441e0,1.24939e0,1.31598e0,1.34792e0),
	    etaHighm2 = cms.vdouble(1.25,1.1,0.95,0.835),
	    etaHighm1 = cms.vdouble(0.85,0.725,0.615,0.54),
	    etaHigh0 = cms.vdouble(0.3,0.25,0.21,0.175),
	    etaHigh1 = cms.vdouble(-0.325,-0.25,-0.225,-0.19),
	    etaHigh2 = cms.vdouble(-0.815,-0.72,-0.6,-0.525),
	    etaLowm2 = cms.vdouble(0.815,0.72,0.6,0.525),
	    etaLowm1 = cms.vdouble(0.325,0.25,0.225,0.19),
	    etaLow0 = cms.vdouble(-0.3,-0.25,-0.21,-0.175),
	    etaLow1 = cms.vdouble(-0.825,-0.725,-0.615,-0.54),
	    etaLow2 = cms.vdouble(-1.25,-1.1,-0.95,-0.835),
	    alpha = cms.vdouble(2.42913e-3,3.03943e-3,4.12679e-3,5.17667e-3),
	    beta = cms.vdouble(5.79143e2,6.08130e2,5.96280e2,1.05578e3)
        )
        
    )
)
