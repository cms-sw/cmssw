# The following comments couldn't be translated into the new config version:

# zeeHLT.cff #########

import FWCore.ParameterSet.Config as cms

report = cms.EDFilter("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults")
)

zeeHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1Electron', 'HLT2Electron'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults")
)


