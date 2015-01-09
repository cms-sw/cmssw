import FWCore.ParameterSet.Config as cms

# Module for 4D rechit building 
# The block of the reconstruction algo
from RecoLocalMuon.DTSegment.DTMeantimerPatternReco4DAlgo_LinearDriftFromDB_cfi import *
dt4DSegments = cms.EDProducer("DTRecSegment4DProducer",
    # The reconstruction algo and its parameter set
    DTMeantimerPatternReco4DAlgo_LinearDriftFromDB,
    # debug option
    debug = cms.untracked.bool(False),
    # name of the rechit 1D collection in the event
    recHits1DLabel = cms.InputTag("dt1DRecHits"),
    # name of the rechit 2D collection in the event
    recHits2DLabel = cms.InputTag("dt2DSegments")
)

#add cosmics reconstruction in collisions
from RecoLocalMuon.DTSegment.DTMeantimerPatternReco4DAlgo_LinearDriftFromDB_CosmicData_cfi import *
dt4DCosmicSegments = cms.EDProducer("DTRecSegment4DProducer",
    # The reconstruction algo and its parameter set
    DTMeantimerPatternReco4DAlgo_LinearDriftFromDB_CosmicData,
    # debuggin opt
    debug = cms.untracked.bool(False),
    # name of the rechit 1D collection in the event
    recHits1DLabel = cms.InputTag("dt1DCosmicRecHits"),
    # name of the rechit 2D collection in the event
    recHits2DLabel = cms.InputTag("dt2DCosmicSegments")
)
