# The following comments couldn't be translated into the new config version:

# zeeHLT.cff #########

import FWCore.ParameterSet.Config as cms

report = cms.EDFilter("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults")
)

zeeHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT_IsoEle15_L1I', 
        'HLT_DoubleIsoEle10_L1I'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults")
)


