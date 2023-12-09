import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
MUOJME = hlt.hltHighLevel.clone()
MUOJME.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
MUOJME.HLTPaths = cms.vstring(
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8*',
    )
MUOJME.andOr = cms.bool( True )
# we want to intentionally throw and exception
# in case it does not match one of the HLT Paths
MUOJME.throw = cms.bool( False )
