# The following comments couldn't be translated into the new config version:

#     bool byName = true

import FWCore.ParameterSet.Config as cms

gammajetHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT_IsoPhoton30_L1I', 
        'HLT_IsoPhoton12_L1I'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


