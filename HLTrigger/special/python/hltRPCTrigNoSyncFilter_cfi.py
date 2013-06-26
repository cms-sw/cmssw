import FWCore.ParameterSet.Config as cms

hltRPCTrigNoSyncFilter = cms.EDAnalyzer('HLTRPCTrigNoSyncFilter',
   GMTInputTag = cms.InputTag("hltGtDigis"),
   rpcRecHits = cms.InputTag("hltRpcRecHits"),
   saveTags = cms.bool( False )
)
