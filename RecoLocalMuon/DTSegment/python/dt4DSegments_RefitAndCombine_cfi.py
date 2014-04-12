import FWCore.ParameterSet.Config as cms

# Module for 4D rechit building 
# The block of the reconstruction algo
from RecoLocalMuon.DTSegment.DTRefitAndCombineReco4DAlgo_cfi import *
dt4DSegments = cms.EDProducer("DTRecSegment4DProducer",
    # The reconstruction algo and its parameter set
    DTRefitAndCombineReco4DAlgo,
    # debuggin opt
    debug = cms.untracked.bool(False),
    # name of the rechit 2D collection in the event
    recHits1DLabel = cms.InputTag("dt1DRecHits"),
    # name of the rechit 2D collection in the event
    recHits2DLabel = cms.InputTag("dt2DSegments")
)


