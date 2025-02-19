import FWCore.ParameterSet.Config as cms

# 
# This is the include file with the parameters
# for the DTMeantimerPatternReco algorithm,
# which is the concrete algo for the DTRecSegment2D production.
# The linear Drift algos is used.
#
# The reconstruction algo and its parameter set
from RecoLocalMuon.DTRecHit.DTLinearDriftAlgo_cfi import *
DTMeantimerPatternReco2DAlgo_LinearDrift = cms.PSet(
    Reco2DAlgoConfig = cms.PSet(
        # Parameters for the updator
        # this is the RecHit1D algo!!
        DTLinearDriftAlgo,
        segmCleanerMode = cms.int32(1),
        AlphaMaxPhi = cms.double(1.0),
        MaxChi2 = cms.double(8.0),
        MaxT0 = cms.double(50.0),
        MaxAllowedHits = cms.uint32(50),
        # Parameters for the cleaner
        nSharedHitsMax = cms.int32(2),
        AlphaMaxTheta = cms.double(0.1),
        debug = cms.untracked.bool(False),
        nUnSharedHitsMin = cms.int32(2),
        MinT0 = cms.double(-10.0)
    ),
    Reco2DAlgoName = cms.string('DTMeantimerPatternReco')
)

