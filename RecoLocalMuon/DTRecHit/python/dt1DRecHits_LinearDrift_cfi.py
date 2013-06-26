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

# add cosmics reconstruction in collisions
from RecoLocalMuon.DTRecHit.DTLinearDriftAlgo_CosmicData_cfi import *
dt1DCosmicRecHits = cms.EDProducer("DTRecHitProducer",
    # The reconstruction algo and its parameter set
    DTLinearDriftAlgo_CosmicData,
    debug = cms.untracked.bool(False),
    # The label to retrieve digis from the event
    #dtDigiLabel = cms.InputTag("dtunpacker")
    dtDigiLabel = cms.InputTag("muonDTDigis")
)
