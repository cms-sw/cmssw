import FWCore.ParameterSet.Config as cms

# Module for reconstruction of cosmic data: reconstruction is performed using constant drift velocity 
# and digi time sinchronization reads ttrig and t0 from DB
# The reconstruction algo and its parameter set
from RecoLocalMuon.DTRecHit.DTLinearDriftAlgo_CosmicData_cfi import *
dt1DRecHits = cms.EDProducer("DTRecHitProducer",
    # The reconstruction algo and its parameter set
    DTLinearDriftAlgo_CosmicData,
    debug = cms.untracked.bool(False),
    # The label to retrieve digis from the event
    #dtDigiLabel = cms.InputTag("dtunpacker")
    dtDigiLabel = cms.InputTag("muonDTDigis")
)


