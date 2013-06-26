import FWCore.ParameterSet.Config as cms

# 
# Module for Jet trigger skim -- HLT1jet trigger
#
QCDHLT1jetTrigger = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1jet'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults")
)


