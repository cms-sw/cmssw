# The following comments couldn't be translated into the new config version:

#     bool byName = true

import FWCore.ParameterSet.Config as cms

gammajetHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1Photon', 
        'HLT1PhotonL1Isolated'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


