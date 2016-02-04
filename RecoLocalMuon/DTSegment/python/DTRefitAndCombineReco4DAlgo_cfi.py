import FWCore.ParameterSet.Config as cms

# this is the RecHit1D algo include!
from RecoLocalMuon.DTRecHit.DTParametrizedDriftAlgo_cfi import *
DTRefitAndCombineReco4DAlgo = cms.PSet(
    Reco4DAlgoName = cms.string('DTRefitAndCombineReco4D'),
    Reco4DAlgoConfig = cms.PSet(
        # Parameters for the updator
        # this is the RecHit1D algo!!
        DTParametrizedDriftAlgo,
        debug = cms.untracked.bool(False),
        # Parameters for the cleaner
        nSharedHitsMax = cms.int32(2),
        MaxChi2forPhi = cms.double(100.0) ##FIX this value!

    )
)

