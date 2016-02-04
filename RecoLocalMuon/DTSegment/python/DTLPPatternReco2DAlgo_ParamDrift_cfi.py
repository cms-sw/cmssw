import FWCore.ParameterSet.Config as cms

# 
# This is the include file with the parameters
# for the DTLPPatternReco algorithm,
# which is the concrete algo for the DTRecSegment2D production.
# The Parametrized drift algo is used.
#
# The reconstruction algo and its parameter set (it is the RecHit1D algo)
from RecoLocalMuon.DTRecHit.DTParametrizedDriftAlgo_cfi import DTParametrizedDriftAlgo
#the recalgo, needed by the updator

DTLPPatternReco2DAlgo_ParamDrift = cms.PSet(
    Reco2DAlgoConfig = cms.PSet(
        # Parameters for the updator
        # this is the RecHit1D algo!!
        DTParametrizedDriftAlgo,
        performT0SegCorrection = cms.bool(False),
        hit_afterT0_resolution = cms.double(0.03),
        performT0_vdriftSegCorrection = cms.bool(False),
        debug = cms.untracked.bool(False),
        #these are the parameters for the 2D algo
        DeltaFactor = cms.double(4.0),
        maxAlphaTheta = cms.double(0.1),
        maxAlphaPhi = cms.double(1.0),
        min_q = cms.double(-300.0),
        max_q = cms.double(300.0),
        bigM = cms.double(100.0),
        ),
    Reco2DAlgoName = cms.string('DTLPPatternReco')
)



