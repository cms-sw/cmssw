# The following comments couldn't be translated into the new config version:

# The reconstruction algo and its parameter set

import FWCore.ParameterSet.Config as cms

# Module for rechit building of simulated digis using the cell 
# parametrization
# The reconstruction algo and its parameter set
from RecoLocalMuon.DTRecHit.DTParametrizedDriftAlgo_cfi import *
dt1DRecHits = cms.EDProducer("DTRecHitProducer",
    DTParametrizedDriftAlgo,
    debug = cms.untracked.bool(False),
    dtDigiLabel = cms.InputTag("muonDTDigis")
)


