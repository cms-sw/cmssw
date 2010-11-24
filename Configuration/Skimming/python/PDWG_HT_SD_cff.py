import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.triggerResultsFilter_cfi as hlt
HTSD = hlt.triggerResultsFilter.clone()
HTSD.triggerConditions = cms.vstring('HLT_HT*',)
HTSD.hltResults = cms.InputTag( "TriggerResults", "", "HLT" )
HTSD.l1tResults = cms.InputTag("")
HTSD.throw = cms.bool( False )
