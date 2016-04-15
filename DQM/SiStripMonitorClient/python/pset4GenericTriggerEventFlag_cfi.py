import FWCore.ParameterSet.Config as cms

genericTriggerEventFlag4fullTrackerAndHLTdb = cms.PSet(
   andOr         = cms.bool( False ),
   dbLabel       = cms.string("TrackerDQMTrigger"),
   andOrHlt      = cms.bool(True), # True:=OR; False:=AND
   hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
   hltPaths      = cms.vstring(""), #cms.vstring("HLT_ZeroBias_v*","HLT_HIZeroBias_v*")
   hltDBKey      = cms.string("SiStrip_HLT"),
   errorReplyHlt = cms.bool( False ),
)
