import FWCore.ParameterSet.Config as cms




## Trigger for MC
import HLTrigger.HLTfilters.hltHighLevel_cfi
TriggerFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
    #HLTPaths = ['HLT_Mu9', 'HLT_Mu15_v*'],
    #HLTPaths = ['HLT_IsoMu17_v*'],
    HLTPaths = ['HLT_IsoMu24_*'], #  # provide list of HLT paths (or patterns) you want
    #HLTPaths = ['@'],
    #andOr = cms.bool(True),   # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
    throw = False, ## throw exception on unknown path names
)



## SEQUENCE
TriggerSelectionSequence = cms.Sequence(
    TriggerFilter
)





