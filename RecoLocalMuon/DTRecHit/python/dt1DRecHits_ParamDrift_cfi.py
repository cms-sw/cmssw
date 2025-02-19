import FWCore.ParameterSet.Config as cms

# Module for rechit building of simulated digis using the cell 
# parametrization
# The reconstruction algo and its parameter set
from RecoLocalMuon.DTRecHit.DTParametrizedDriftAlgo_cfi import *
dt1DRecHits = cms.EDProducer("DTRecHitProducer",
    # The reconstruction algo and its parameter set
    DTParametrizedDriftAlgo,
    debug = cms.untracked.bool(False),
    dtDigiLabel = cms.InputTag("muonDTDigis")
)


