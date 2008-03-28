# The following comments couldn't be translated into the new config version:

# The reconstruction algo and its parameter set

import FWCore.ParameterSet.Config as cms

# Module for reconstruction of simulated data for CSA07: reconstruction is performed using fake drift velocity
# from DB and digi time sinchronization reads fake ttrig and fake t0 from DB
# The reconstruction algo and its parameter set
from RecoLocalMuon.DTRecHit.DTLinearDriftFromDBAlgo_cfi import *
dt1DRecHits = cms.EDProducer("DTRecHitProducer",
    DTLinearDriftFromDBAlgo,
    debug = cms.untracked.bool(True),
    # The label to retrieve digis from the event
    dtDigiLabel = cms.InputTag("muonDTDigis")
)


