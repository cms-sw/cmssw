import FWCore.ParameterSet.Config as cms

triggerResultsFilter = cms.EDFilter('TriggerResultsFilter',
    hltResults    = cms.InputTag('TriggerResults','','HLT'),    # HLT results
    l1tResults    = cms.InputTag('gtDigis'),                    # L1 GT results
    l1tIgnoreMask = cms.bool(False),                            # use L1 mask
    throw         = cms.bool(True),                             # throw exception on unknown path names
    triggerConditions = cms.vstring(
        'HLT_*',
    )
)
