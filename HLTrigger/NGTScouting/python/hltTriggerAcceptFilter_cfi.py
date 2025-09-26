import FWCore.ParameterSet.Config as cms
from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as _triggerResultsFilter

hltTriggerAcceptFilter = _triggerResultsFilter.clone(
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( False ),
    triggerConditions = cms.vstring('HLT_*', 'MC_*')
)

dstTriggerAcceptFilter = _triggerResultsFilter.clone(
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( False ),
    triggerConditions = cms.vstring('DST_*')
)
