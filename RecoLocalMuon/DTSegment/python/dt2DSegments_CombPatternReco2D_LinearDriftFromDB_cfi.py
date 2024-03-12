import FWCore.ParameterSet.Config as cms

# Module for 2D rechit building 
# The algo uses the Linear Drift Velocity from DB (which is a 1D rec hit algo)
# The block of the reconstruction
from RecoLocalMuon.DTSegment.DTCombinatorialPatternReco2DAlgo_LinearDriftFromDB_cfi import *
dt2DSegments = cms.EDProducer("DTRecSegment2DProducer",
    # The reconstruction algo and its parameter set
    DTCombinatorialPatternReco2DAlgo_LinearDriftFromDB,
    # debuggin opt
    debug = cms.untracked.bool(False),
    # name of the rechit 1D collection in the event
    recHits1DLabel = cms.InputTag("dt1DRecHits")
)

#add cosmics reconstruction in collisions
from RecoLocalMuon.DTSegment.DTCombinatorialPatternReco2DAlgo_LinearDriftFromDB_cfi import *
dt2DCosmicSegments = cms.EDProducer("DTRecSegment2DProducer",
    # The reconstruction algo and its parameter set
    DTCombinatorialPatternReco2DAlgo_LinearDriftFromDB,
    # debuggin opt
    debug = cms.untracked.bool(False),
    # name of the rechit 1D collection in the event
    recHits1DLabel = cms.InputTag("dt1DCosmicRecHits")
)
# foo bar baz
# LdEwEOJDrCAg3
# S5wdIwRyh0u7D
