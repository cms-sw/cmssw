import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
CSCClusterTrigger = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
CSCClusterTrigger.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
CSCClusterTrigger.HLTPaths = cms.vstring(
    ["*CscCluster*","*L1CSCShower*"]
)
CSCClusterTrigger.throw = False
CSCClusterTrigger.andOr = True

EXOCSCClusterSkimSequence = cms.Sequence(
    CSCClusterTrigger
)
