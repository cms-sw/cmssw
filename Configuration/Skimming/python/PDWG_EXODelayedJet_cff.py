import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
DelayedJetHTTrigger = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
DelayedJetHTTrigger.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
DelayedJetHTTrigger.HLTPaths = cms.vstring(
    ["HLT_HT430_DelayedJet40*","HLT_HT350_DelayedJet40*","HLT_L1Tau_DelayedJet40*"]
)
DelayedJetHTTrigger.throw = False
DelayedJetHTTrigger.andOr = True

EXODelayedJetSkimSequence = cms.Sequence(
    DelayedJetHTTrigger
)
