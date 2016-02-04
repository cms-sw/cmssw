import FWCore.ParameterSet.Config as cms

# Module for reconstruction of simulated data for CSA07: reconstruction is performed using fake drift velocity
# from DB and digi time sinchronization reads fake ttrig and fake t0 from DB
# The reconstruction algo and its parameter set
from RecoLocalMuon.DTRecHit.DTLinearDriftFromDBAlgo_cfi import *
dt1DRecHits = cms.EDProducer("DTRecHitProducer",
    # The reconstruction algo and its parameter set
    DTLinearDriftFromDBAlgo,
    debug = cms.untracked.bool(False),
    # The label to retrieve digis from the event
    dtDigiLabel = cms.InputTag("muonDTDigis")
)


# Also add the cosmics reconstruction in collisions
from RecoLocalMuon.DTRecHit.DTLinearDriftFromDBAlgo_CosmicData_cfi import *
dt1DCosmicRecHits = cms.EDProducer("DTRecHitProducer",
    # The reconstruction algo and its parameter set
    DTLinearDriftFromDBAlgo_CosmicData,
    debug = cms.untracked.bool(False),
    # The label to retrieve digis from the event
    #dtDigiLabel = cms.InputTag("dtunpacker")
    dtDigiLabel = cms.InputTag("muonDTDigis")
)
