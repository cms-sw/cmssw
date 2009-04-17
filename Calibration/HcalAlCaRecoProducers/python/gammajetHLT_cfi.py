# The following comments couldn't be translated into the new config version:

#     bool byName = true

import FWCore.ParameterSet.Config as cms

gammajetHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT_IsoPhoton30_L1I', 
        'HLT_IsoPhoton15_L1R'),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False), #dont throw except on unknown path name 
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


