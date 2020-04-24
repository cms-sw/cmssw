import FWCore.ParameterSet.Config as cms

genericTriggerEventFlag4HLTdb = cms.PSet(
   andOr         = cms.bool( False ),
   dbLabel       = cms.string("SiStripDQMTrigger"), # ("TrackerDQMTrigger"),
   andOrHlt      = cms.bool(True), # True:=OR; False:=AND
   hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
   hltPaths      = cms.vstring("HLT_ZeroBias_v*","HLT_HIZeroBias_v*","HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_*"),
   hltDBKey      = cms.string("SiStrip_HLT"),
   errorReplyHlt = cms.bool( False ),
   verbosityLevel = cms.uint32(1)
)

genericTriggerEventFlag4L1bd = cms.PSet(
   andOr         = cms.bool( False ),
   dbLabel       = cms.string("SiStripDQMTrigger"), # ("TrackerDQMTrigger"),
   l1Algorithms  = cms.vstring(""), # cms.vstring( 'L1Tech_BPTX_plus_AND_minus.v0', 'L1_ZeroBias' )
   andOrL1       = cms.bool( True ),
#   errorReplyL1  = cms.bool( True ),
   errorReplyL1  = cms.bool( False ),
   l1DBKey       = cms.string("SiStrip_L1"),
   l1BeforeMask  = cms.bool( True ), # specifies, if the L1 algorithm decision should be read as before (true) or after (false) masking is applied.
   verbosityLevel = cms.uint32(1)
)
