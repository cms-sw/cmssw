import FWCore.ParameterSet.Config as cms

# 
# This is the include file with the parameters
# for the DTCombinatorialPatternReco algorithm,
# which is the concrete algo for the DTRecSegment2D production.
# The linear Drift algos is used.
#
# The reconstruction algo and its parameter set
from RecoLocalMuon.DTRecHit.DTLinearDriftAlgo_CosmicData_cfi import *
DTCombinatorialPatternReco2DAlgo_LinearDrift_CosmicData = cms.PSet(
    Reco2DAlgoConfig = cms.PSet(
        # Parameters for the updator
        # this is the RecHit1D algo!!
        DTLinearDriftAlgo_CosmicData,
        AlphaMaxPhi = cms.double(100.0),
        AlphaMaxTheta = cms.double(100.0),
        MaxAllowedHits = cms.uint32(50),
        debug = cms.untracked.bool(False),

        # Parameters for the cleaner
        segmCleanerMode = cms.int32(1),
        nSharedHitsMax = cms.int32(2),
        nUnSharedHitsMin = cms.int32(2),

        # Parameters for the Updator and T0 fit segment 
        performT0_vdriftSegCorrection = cms.bool(False),
        hit_afterT0_resolution = cms.double(0.03),
        performT0SegCorrection = cms.bool(False)
    ),
    Reco2DAlgoName = cms.string('DTCombinatorialPatternReco')
)

