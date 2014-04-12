import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.triggerResultsFilter_cfi as hlt
HLTZEROBIASPUSD = hlt.triggerResultsFilter.clone()
HLTZEROBIASPUSD.triggerConditions = cms.vstring('HLT_ZeroBias_v*',)
HLTZEROBIASPUSD.hltResults = cms.InputTag( "TriggerResults", "", "HLT" )
HLTZEROBIASPUSD.l1tResults = cms.InputTag("")
#HLTZEROBIASPUSD.throw = cms.bool( False )

