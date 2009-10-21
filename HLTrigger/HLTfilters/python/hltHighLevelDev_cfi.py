import FWCore.ParameterSet.Config as cms

hltHighLevelDev = cms.EDFilter('HLTHighLevelDev',
    TriggerResultsTag = cms.InputTag('TriggerResults','','HLT'),
    HLTPaths = cms.vstring(),               # provide the list of HLT paths (or patterns) you want
    HLTPathsPrescales = cms.vuint32(),      # provide the list of prescales correseponding to the paths and patterns
    HLTOverallPrescale = cms.uint32(1),     # privide the overall prescale used on top of the final result
    eventSetupPathsKey = cms.string(''),    # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
    andOr = cms.bool(True),                 # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
    throw = cms.bool(True)                  # throw exception on unknown path names
)
