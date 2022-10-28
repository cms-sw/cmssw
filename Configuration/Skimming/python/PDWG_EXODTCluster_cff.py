import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
DTClusterHTTrigger = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
DTClusterHTTrigger.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
DTClusterHTTrigger.HLTPaths = cms.vstring(
    "*DTCluster*"
)
DTClusterHTTrigger.throw = False
DTClusterHTTrigger.andOr = True

EXODTClusterSkimSequence = cms.Sequence(
    DTClusterHTTrigger
)
