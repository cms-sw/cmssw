import FWCore.ParameterSet.Config as cms

# 
# This is the include file with the parameters
# for the DTCombinatorialPatternReco algorithm,
# which is the concrete algo for the DTRecSegment4D production.
# The Parametrized Drift algos is used.
#
# this is the RecSegment2D algo include!
from RecoLocalMuon.DTSegment.DTLPPatternReco2DAlgo_ParamDrift_cfi import *
# this is the RecHit1D algo include!
from RecoLocalMuon.DTRecHit.DTParametrizedDriftAlgo_cfi import *
DTLPPatternReco4DAlgo_ParamDrift = cms.PSet(
    Reco4DAlgoName = cms.string('DTLPPatternReco4D'),
    Reco4DAlgoConfig = cms.PSet(
        # this are the RecSegment2D algo parameters!
        DTLPPatternReco2DAlgo_ParamDrift,
        # Parameters for the updator
        # this is the RecHit1D algo!!
        DTParametrizedDriftAlgo,
        segmCleanerMode = cms.int32(1),
        # Parameters for the cleaner
        nSharedHitsMax = cms.int32(2),
        debug = cms.untracked.bool(False),
        nUnSharedHitsMin = cms.int32(2),
        # the input type. 
        # If true the instructions in setDTRecSegment2DContainer will be skipped and the 
        # theta segment will be recomputed from the 1D rechits
        # If false the theta segment will be taken from the Event. Caveat: in this case the
        # event must contain the 2D segments!
        AllDTRecHits = cms.bool(True),
        # Parameters for  T0 fit segment in the Updator 
        performT0SegCorrection = cms.bool(False),
        hit_afterT0_resolution = cms.double(0.03),
        performT0_vdriftSegCorrection = cms.bool(False)
    )
)


