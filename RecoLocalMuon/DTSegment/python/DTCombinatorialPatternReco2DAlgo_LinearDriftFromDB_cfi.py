import FWCore.ParameterSet.Config as cms

# 
# This is the include file with the parameters
# for the DTCombinatorialPatternReco algorithm,
# which is the concrete algo for the DTRecSegment2D production.
# The linear Drift algo which read from DB is used.
#
# The reconstruction algo and its parameter set
from RecoLocalMuon.DTRecHit.DTLinearDriftFromDBAlgo_cfi import *
DTCombinatorialPatternReco2DAlgo_LinearDriftFromDB = cms.PSet(
    Reco2DAlgoConfig = cms.PSet(
        # Parameters for the updator
        # this is the RecHit1D algo!!
        DTLinearDriftFromDBAlgo,
        segmCleanerMode = cms.int32(2),
        AlphaMaxPhi = cms.double(1.0),
        MaxAllowedHits = cms.uint32(50),
        # Parameters for the cleaner
        nSharedHitsMax = cms.int32(2),
        AlphaMaxTheta = cms.double(0.9),
        debug = cms.untracked.bool(False),
        nUnSharedHitsMin = cms.int32(2),
        # Parameters for  T0 fit segment in the Updator 
        performT0_vdriftSegCorrection = cms.bool(False),
        hit_afterT0_resolution = cms.double(0.03),
        performT0SegCorrection = cms.bool(False),
        perform_delta_rejecting = cms.bool(True)
    ),
    Reco2DAlgoName = cms.string('DTCombinatorialPatternReco')
)

