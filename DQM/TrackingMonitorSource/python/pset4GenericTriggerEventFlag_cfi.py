import FWCore.ParameterSet.Config as cms

genericTriggerEventFlag4fullTracker = cms.PSet(
   andOr         = cms.bool( False ),
   dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
   dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ),
   andOrDcs      = cms.bool( False ),
   errorReplyDcs = cms.bool( True ),
)
genericTriggerEventFlag4onlyStrip = cms.PSet(
   andOr         = cms.bool( False ),
   dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
   dcsPartitions = cms.vint32 ( 24, 25, 26, 27 ),
   andOrDcs      = cms.bool( False ),
   errorReplyDcs = cms.bool( True ),
)
genericTriggerEventFlag4fullTrackerAndHLTdb = cms.PSet(
   andOr         = cms.bool( False ),
   dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
   dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ),
   andOrDcs      = cms.bool( False ),
   errorReplyDcs = cms.bool( True ),
   dbLabel       = cms.string("SiStripDQMTrigger"), #("TrackerDQMTrigger"),
   andOrHlt      = cms.bool(True),# True:=OR; False:=AND
   hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
   hltPaths      = cms.vstring(""), # HLT_ZeroBias_v*
   hltDBKey      = cms.string("Tracking_HLT"),
   errorReplyHlt = cms.bool( False ),
   verbosityLevel = cms.uint32(1)
)
