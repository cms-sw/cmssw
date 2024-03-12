import FWCore.ParameterSet.Config as cms

# 
# This is the include file with the parameters
# for the DTCombinatorialPatternReco algorithm,
# which is the concrete algo for the DTRecSegment2D production.
# The linear Drift algos is used.
#
# The reconstruction algo and its parameter set
from RecoLocalMuon.DTRecHit.DTNoDriftAlgo_CosmicData_cfi import *
DTCombinatorialPatternReco2DAlgo_NoDrift_CosmicData = cms.PSet(
    Reco2DAlgoConfig = cms.PSet(
        # Parameters for the updator
        # this is the RecHit1D algo!!
        DTNoDriftAlgo_CosmicData,
        segmCleanerMode = cms.int32(1),
        AlphaMaxPhi = cms.double(100.0),
        MaxAllowedHits = cms.uint32(50),
        # Parameters for the cleaner
        nSharedHitsMax = cms.int32(2),
        AlphaMaxTheta = cms.double(100.0),
        debug = cms.untracked.bool(False),
        nUnSharedHitsMin = cms.int32(2),
        # Parameters for  T0 fit segment in the Updator and
        performT0SegCorrection = cms.bool(False),
        performT0_vdriftSegCorrection = cms.bool(False),
        hit_afterT0_resolution = cms.double(0.03)
    ),
    Reco2DAlgoName = cms.string('DTCombinatorialPatternReco')
)

# foo bar baz
# vbL01hy84ITHx
# spDvpL5bD5TdQ
