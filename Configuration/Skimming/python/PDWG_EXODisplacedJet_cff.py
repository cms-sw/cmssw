import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
DisplacedJetHTTrigger = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
DisplacedJetHTTrigger.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
DisplacedJetHTTrigger.HLTPaths = cms.vstring(
    "HLT_HT425_v*"
)
DisplacedJetHTTrigger.throw = False
DisplacedJetHTTrigger.andOr = True

EXODisplacedJetSkimSequence = cms.Sequence(
    DisplacedJetHTTrigger
)
