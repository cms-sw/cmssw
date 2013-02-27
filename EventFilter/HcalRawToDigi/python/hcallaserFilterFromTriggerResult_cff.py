import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel

hcalfilter = hltHighLevel.clone(
    TriggerResultsTag = cms.InputTag('TriggerResults'),
    HLTPaths = cms.vstring('user_step' )
    )

hcalfilterSeq = cms.Sequence( hcalfilter )
