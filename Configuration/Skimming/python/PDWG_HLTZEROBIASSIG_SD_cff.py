import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.triggerResultsFilter_cfi as hlt
HLTZEROBIASSIGSD = hlt.triggerResultsFilter.clone()
HLTZEROBIASSIGSD.triggerConditions = cms.vstring('HLT_Physics_v*',)
HLTZEROBIASSIGSD.hltResults = cms.InputTag( "TriggerResults", "", "HLT" )
HLTZEROBIASSIGSD.l1tResults = cms.InputTag("")
#HLTZEROBIASSIGSD.throw = cms.bool( False )
