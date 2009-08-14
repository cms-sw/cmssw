import FWCore.ParameterSet.Config as cms

# 
# This is the include file with the parameters
# for the DTLPPatternReco algorithm,
# which is the concrete algo for the DTRecSegment2D production.
# The Parametrized drift algo is used.
#
# The reconstruction algo and its parameter set (it is the RecHit1D algo)
from RecoLocalMuon.DTRecHit.DTParametrizedDriftAlgo_cfi import DTParametrizedDriftAlgo

DTLPPatternReco2DAlgo_ParamDrift = cms.PSet(
    Reco2DAlgoConfig = cms.PSet(
        # Parameters for the updator
        # this is the RecHit1D algo!!
        DTParametrizedDriftAlgo,
        DeltaFactor = cms.double(4.0),
        min_m = cms.double(-10),
        max_m = cms.double(10),
        min_q = cms.double(-100.0),
        max_q = cms.double(100.0),
        bigM = cms.double(100.0),
    ),
    Reco2DAlgoName = cms.string('DTLPPatternReco')
)



