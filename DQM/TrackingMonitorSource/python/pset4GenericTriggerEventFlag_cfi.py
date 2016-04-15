import FWCore.ParameterSet.Config as cms

genericTriggerEventFlag4fullTrackerAndHLTdb = cms.PSet(
   andOr         = cms.bool( False ),
   dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
   dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ),
   andOrDcs      = cms.bool( False ),
   errorReplyDcs = cms.bool( True ),
   dbLabel       = cms.string("TrackerDQMTrigger"),
   andOrHlt      = cms.bool(True),
   hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
   hltPaths      = cms.vstring(""),
   hltDBKey      = cms.string("Tracking_HLT"),
   errorReplyHlt = cms.bool( False ),
)
