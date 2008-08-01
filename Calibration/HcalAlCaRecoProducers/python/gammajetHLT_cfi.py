# The following comments couldn't be translated into the new config version:

#     bool byName = true

import FWCore.ParameterSet.Config as cms

gammajetHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT_IsoPhoton30_L1I', 
        'HLT_IsoPhoton15_L1R'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


