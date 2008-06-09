import FWCore.ParameterSet.Config as cms

# Module for rechit building of simulated digis using the constant 
# drift velocity over the entire cell
# The reconstruction algo and its parameter set
from RecoLocalMuon.DTRecHit.DTLinearDriftAlgo_cfi import *
dt1DRecHits = cms.EDProducer("DTRecHitProducer",
    # The reconstruction algo and its parameter set
    DTLinearDriftAlgo,
    debug = cms.untracked.bool(False),
    dtDigiLabel = cms.InputTag("muonDTDigis")
)


